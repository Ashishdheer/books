"""
radar_sim_v2.py  -  physics-based, noise-free target simulator (2.5 s output grid)

Every target is a point mass flown by a simple autopilot through a *mission* (a chain of behaviours).
Dynamics run on a fine internal step; the radar-style packets are sampled exactly every 2.5 s: t = 0, 2.5, 5, ...
No measurement noise and no tracker in this version: position, velocity and acceleration are the TRUE values.

Two motion models
  'fw'  : velocity = (speed V, heading psi, flight-path angle gamma). Controls: speed, turn rate (bank-limited), climb.
          Used for aeroplanes, fixed-wing drones, missiles, boats and ships (ships/boats have gamma = 0).
  'vec' : ground-velocity vector with an acceleration (tilt) limit. Used for helicopters and multirotor drones,
          so they can stop, hover, translate sideways and land.
Axes: x = east, y = north, z = up (metres), radar at the origin.  heading = atan2(vy, vx) in degrees (as in the old file).
"""
import math, random
import numpy as np, pandas as pd

G = 9.81
DT_OUT = 2.5
RADAR_H = 20.0
CLASSES = ["CIVILIAN_AIRCRAFT", "FIGHTER_AIRCRAFT", "HELICOPTER", "DRONE", "SURFACE_MISSILE",
           "AERIAL_MISSILE", "SKIFF", "BOAT", "SHIP"]
ICD_COLS = ["packet_size", "source", "destination", "packet_type", "packet_category", "radar_id", "track_no",
            "pos_x", "pos_y", "pos_z", "range_mag", "vel_x", "vel_y", "vel_z", "vel_mag", "heading",
            "accel_x", "accel_y", "accel_z", "accel_mag", "std_dev_x", "std_dev_y", "std_dev_z",
            "std_dev_vx", "std_dev_vy", "std_dev_vz", "std_dev_ax", "std_dev_ay", "std_dev_az",
            "cov_x_vx", "cov_y_vy", "cov_z_vz", "ntp_sec", "ntp_frac", "assoc_indication"]

def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi
def clip(x, lo, hi): return lo if x < lo else (hi if x > hi else x)
def horizon(z): return 4120.0 * (math.sqrt(RADAR_H) + math.sqrt(max(z, 0.0)))


class Agent:
    def __init__(s, kind, rng, dt, **kw):
        s.kind, s.rng, s.dt = kind, rng, dt
        # state
        s.x = s.y = s.z = 0.0; s.V = 100.0; s.psi = 0.0; s.gam = 0.0
        s.vx = s.vy = s.vz = 0.0                                   # (vec model) ground velocity
        s.al = s.om = s.gd = 0.0; s.ax = s.ay = s.az = 0.0
        s.hdg = 0.0; s.t = 0.0
        # commands
        s.Vc = 100.0; s.psic = 0.0; s.zc = 0.0
        s.al_cmd = s.alat_cmd = s.avert_cmd = None                  # explicit accelerations (missiles)
        s.vcx = s.vcy = 0.0
        # limits / time constants (defaults = airliner-like); overridden by kw
        s.Vmin, s.Vmax = 40.0, 300.0; s.a_acc, s.a_dec = 1.0, 1.5; s.tau_v = 8.0; s.tau_a = 2.0
        s.glat = 0.45; s.om_cap = None; s.tau_om = 3.0; s.tau_psi = 6.0
        s.gam_max = math.radians(25); s.vz_max = 15.0; s.az_max = 1.5; s.tau_g = 4.0; s.tau_pitch = 2.0
        s.planar = False; s.zmin = 0.0; s.drag_k = 0.0; s.grav = True; s.n_lat_g = 0.0
        s.a_max = 3.0; s.tau_vz = 1.5                                 # vec model
        s.wind = (0.0, 0.0); s.wind_ref = 1.0
        s.heave = []; s.z_ref = 0.0
        s.phase = "init"; s.acting = None; s.done = False
        for k, v in kw.items(): setattr(s, k, v)

    # ---------- helpers ----------
    def turn_radius(self):
        om = (self.om_cap if self.om_cap else self.glat * G / max(self.V * math.cos(self.gam), 10.0))
        return max(self.V * math.cos(self.gam), 1.0) / max(om, 1e-6)

    def wind_at(self):
        f = clip(0.25 + self.z / 4000.0, 0.25, 1.0) * self.wind_ref
        return self.wind[0] * f, self.wind[1] * f

    # ---------- dynamics: fixed-wing / missile / ship ----------
    def step_fw(s):
        dt = s.dt; V = max(s.V, 1.0); cg = max(math.cos(s.gam), 0.1)
        if s.al_cmd is not None: al_t = s.al_cmd
        else: al_t = clip((s.Vc - s.V) / s.tau_v, -s.a_dec, s.a_acc)
        if s.grav and not s.planar: al_t -= G * math.sin(s.gam)
        al_t -= s.drag_k * s.n_lat_g ** 2
        s.al += (al_t - s.al) * min(1.0, dt / s.tau_a)
        omax = s.om_cap if s.om_cap else s.glat * G / (V * cg)
        if s.alat_cmd is not None: om_t = clip(s.alat_cmd / (V * cg), -omax, omax)
        else: om_t = clip(wrap(s.psic - s.psi) / s.tau_psi, -omax, omax)
        s.om += (om_t - s.om) * min(1.0, dt / s.tau_om)
        if s.planar: gd_t = 0.0
        elif s.avert_cmd is not None: gd_t = clip(s.avert_cmd / V, -s.az_max / V, s.az_max / V)
        else:
            e = s.zc - s.z
            vz_des = math.copysign(min(s.vz_max, math.sqrt(2 * 0.4 * s.az_max * abs(e)), abs(e) / 6.0 + 0.5), e)
            gam_des = math.asin(clip(vz_des / V, -math.sin(s.gam_max), math.sin(s.gam_max)))
            gd_t = clip((gam_des - s.gam) / s.tau_g, -s.az_max / V, s.az_max / V)
        s.gd += (gd_t - s.gd) * min(1.0, dt / s.tau_pitch)
        s.n_lat_g = math.hypot(V * s.om * cg, V * s.gd) / G
        Vo, po, go = s.V, s.psi, s.gam
        s.V = clip(s.V + s.al * dt, s.Vmin, s.Vmax); s.psi = wrap(s.psi + s.om * dt)
        s.gam = clip(s.gam + s.gd * dt, -math.radians(85), math.radians(85))
        wx, wy = s.wind_at()
        v0 = (Vo * math.cos(go) * math.cos(po), Vo * math.cos(go) * math.sin(po), Vo * math.sin(go))
        v1 = (s.V * math.cos(s.gam) * math.cos(s.psi), s.V * math.cos(s.gam) * math.sin(s.psi), s.V * math.sin(s.gam))
        s.x += 0.5 * (v0[0] + v1[0]) * dt + wx * dt; s.y += 0.5 * (v0[1] + v1[1]) * dt + wy * dt
        s.z += 0.5 * (v0[2] + v1[2]) * dt
        if s.z < s.zmin:
            s.z = s.zmin
            if s.gam < 0: s.gam = 0.0; s.gd = 0.0
        s.t += dt

    def out_fw(s):
        cg, sg, cp, sp = math.cos(s.gam), math.sin(s.gam), math.cos(s.psi), math.sin(s.psi)
        wx, wy = s.wind_at()
        v = (s.V * cg * cp + wx, s.V * cg * sp + wy, s.V * sg)
        a = (s.al * cg * cp - s.V * s.gd * sg * cp - s.V * cg * sp * s.om,
             s.al * cg * sp - s.V * s.gd * sg * sp + s.V * cg * cp * s.om,
             s.al * sg + s.V * s.gd * cg)
        return v, a

    # ---------- dynamics: rotorcraft / multirotor ----------
    def step_vec(s):
        dt = s.dt
        ax_t = (s.vcx - s.vx) / s.tau_v; ay_t = (s.vcy - s.vy) / s.tau_v
        m = math.hypot(ax_t, ay_t)
        if m > s.a_max: ax_t *= s.a_max / m; ay_t *= s.a_max / m
        k = min(1.0, dt / s.tau_a)
        s.ax += (ax_t - s.ax) * k; s.ay += (ay_t - s.ay) * k
        e = s.zc - s.z
        vzc = math.copysign(min(s.vz_max, math.sqrt(2 * 0.5 * s.az_max * abs(e)), abs(e) / 3.0), e)
        az_t = clip((vzc - s.vz) / s.tau_vz, -s.az_max, s.az_max)
        s.az += (az_t - s.az) * k
        vx0, vy0, vz0 = s.vx, s.vy, s.vz
        s.vx += s.ax * dt; s.vy += s.ay * dt; s.vz += s.az * dt
        s.x += 0.5 * (vx0 + s.vx) * dt; s.y += 0.5 * (vy0 + s.vy) * dt; s.z += 0.5 * (vz0 + s.vz) * dt
        if s.z < s.zmin: s.z = s.zmin; s.vz = max(s.vz, 0.0); s.az = max(s.az, 0.0)
        s.t += dt

    def out_vec(s):
        return (s.vx, s.vy, s.vz), (s.ax, s.ay, s.az)

    def step(s):
        s.step_fw() if s.kind == "fw" else s.step_vec()

    def sample(s):
        v, a = s.out_fw() if s.kind == "fw" else s.out_vec()
        x, y, z = s.x, s.y, s.z
        if s.heave:                                          # deterministic sea heave (surface targets)
            for A, T, ph in s.heave:
                w = 2 * math.pi / T
                z += A * math.sin(w * s.t + ph); v = (v[0], v[1], v[2] + A * w * math.cos(w * s.t + ph))
                a = (a[0], a[1], a[2] - A * w * w * math.sin(w * s.t + ph))
        hs = math.hypot(v[0], v[1])
        if hs > 0.5: s.hdg = math.atan2(v[1], v[0])
        return x, y, z, v, a, s.hdg


# =========================================================================================
#  behaviour building blocks (generators: each `yield` advances one fine time step)
# =========================================================================================
def hold(ag, T):
    for _ in range(int(T / ag.dt)): yield

def fw_goto(ag, wx, wy, V, z, tol=400.0, Tmax=1e9):
    tol = max(tol, 1.3 * ag.turn_radius()); t = 0.0
    yield
    while t < Tmax:
        dx, dy = wx - ag.x, wy - ag.y; d = math.hypot(dx, dy)
        if d < tol: return
        ag.psic = math.atan2(dy, dx); ag.Vc = V; ag.zc = z
        yield; t += ag.dt

def fw_leg(ag, psi, T, V, z):
    ag.psic = psi; ag.Vc = V; ag.zc = z
    for _ in range(int(T / ag.dt)): yield

def fw_alt(ag, z, V, T=1e9, tol=40.0):
    ag.psic = ag.psi; ag.Vc = V; ag.zc = z; t = 0.0
    while abs(ag.z - z) > tol and t < T: yield; t += ag.dt

def orbit_dir(x, y, cx, cy, R, ccw):
    rx, ry = x - cx, y - cy; d = max(math.hypot(rx, ry), 1.0); rx /= d; ry /= d
    tx, ty = (-ry, rx) if ccw else (ry, -rx)
    kap = clip((d - R) / (0.6 * R), -1.0, 1.0) * 1.1
    ux = tx * math.cos(kap) - rx * math.sin(kap); uy = ty * math.cos(kap) - ry * math.sin(kap)
    return ux, uy

def fw_orbit(ag, cx, cy, R, ccw, V, z, T):
    R = max(R, 1.15 * ag.turn_radius())
    for _ in range(int(T / ag.dt)):
        ux, uy = orbit_dir(ag.x, ag.y, cx, cy, R, ccw)
        ag.psic = math.atan2(uy, ux); ag.Vc = V; ag.zc = z
        yield

def fw_racetrack(ag, hdg, L, V, z, T, right=True, tol=300.0):
    R = 1.2 * ag.turn_radius(); sgn = -1 if right else 1
    ex, ey = math.cos(hdg), math.sin(hdg); nx, ny = -ey * sgn, ex * sgn
    x0, y0 = ag.x, ag.y; t0 = ag.t
    pts = [(x0 + ex * L, y0 + ey * L), (x0 + ex * L + 2 * R * nx, y0 + ey * L + 2 * R * ny),
           (x0 + 2 * R * nx, y0 + 2 * R * ny), (x0, y0)]
    i = 0
    while ag.t - t0 < T:
        yield from fw_goto(ag, pts[i % 4][0], pts[i % 4][1], V, z, tol=tol, Tmax=T - (ag.t - t0)); i += 1

def vec_set_dir(ag, ux, uy, speed): ag.vcx, ag.vcy = ux * speed, uy * speed

def vec_goto(ag, wx, wy, speed, z, tol=4.0, stop=True, Tmax=1e9):
    t = 0.0
    while t < Tmax:
        dx, dy = wx - ag.x, wy - ag.y; d = math.hypot(dx, dy)
        if stop:
            if d < tol and math.hypot(ag.vx, ag.vy) < 0.8: return
            v = min(speed, math.sqrt(2 * ag.a_max * 0.5 * max(d - 0.5 * tol, 0.0)))
        else:
            if d < tol: return
            v = speed
        vec_set_dir(ag, dx / max(d, 1e-6), dy / max(d, 1e-6), v); ag.zc = z
        yield; t += ag.dt

def vec_orbit(ag, cx, cy, R, ccw, speed, z, T):
    for _ in range(int(T / ag.dt)):
        ux, uy = orbit_dir(ag.x, ag.y, cx, cy, R, ccw)
        vec_set_dir(ag, ux, uy, speed); ag.zc = z
        yield

def vec_hover(ag, T, amp=None, alt_amp=0.3):
    """Station keeping: hold a point with a small, slow, deterministic wander (gust response / rotor wash)."""
    r = ag.rng; amp = amp if amp is not None else r.uniform(0.15, 0.7)
    comps = [(amp * r.uniform(0.5, 1.0), r.uniform(0.05, 0.45), r.uniform(0, 6.28)) for _ in range(3)]
    compsy = [(amp * r.uniform(0.5, 1.0), r.uniform(0.05, 0.45), r.uniform(0, 6.28)) for _ in range(3)]
    hx, hy, hz = ag.x, ag.y, ag.zc if ag.zc > 0 else ag.z; ph = r.uniform(0, 6.28); t0 = ag.t
    for _ in range(int(T / ag.dt)):
        tt = ag.t - t0
        wx = sum(a * math.sin(w * tt + p) for a, w, p in comps) / 2.0
        wy = sum(a * math.sin(w * tt + p) for a, w, p in compsy) / 2.0
        ag.vcx = clip(0.4 * (hx - ag.x), -2, 2) + wx; ag.vcy = clip(0.4 * (hy - ag.y), -2, 2) + wy
        ag.zc = hz + alt_amp * math.sin(0.3 * tt + ph)
        yield

def vec_vertical(ag, z, rate, hold_xy=True):
    ag.vz_max = rate; hx, hy = ag.x, ag.y
    while abs(ag.z - z) > 1.5:
        if hold_xy: ag.vcx = clip(0.4 * (hx - ag.x), -2, 2); ag.vcy = clip(0.4 * (hy - ag.y), -2, 2)
        ag.zc = z; yield


# =========================================================================================
#  class factories: each returns (agent, mission generator, meta dict)
# =========================================================================================
rad = math.radians

def place(ag, rng, r_lo, r_hi, mode=None):
    r0 = rng.uniform(r_lo, max(r_hi, r_lo + 1000.0)); th = rng.uniform(0, 2 * math.pi)
    ag.x, ag.y = r0 * math.cos(th), r0 * math.sin(th)
    mode = mode or rng.choice(["inbound", "crossing", "outbound", "random"])
    base = math.atan2(-ag.y, -ag.x)
    off = {"inbound": rng.gauss(0, 0.25), "outbound": math.pi + rng.gauss(0, 0.25),
           "crossing": rng.choice([-1, 1]) * (math.pi / 2 + rng.gauss(0, 0.3)), "random": rng.uniform(-math.pi, math.pi)}[mode]
    return wrap(base + off), mode

def set_air_wind(ag, rng, vmax):
    sp = rng.uniform(0, vmax); d = rng.uniform(0, 2 * math.pi); ag.wind = (sp * math.cos(d), sp * math.sin(d))

def fw_init(ag, psi, V, z):
    ag.psi = ag.psic = psi; ag.V = ag.Vc = V; ag.z = ag.zc = z; ag.gam = 0.0; ag.hdg = psi

def vec_init(ag, psi, V, z):
    ag.vx, ag.vy = V * math.cos(psi), V * math.sin(psi); ag.vcx, ag.vcy = ag.vx, ag.vy; ag.z = ag.zc = z; ag.hdg = psi

def next_wp(ag, rng, L, dmax_deg, dpref=22.0):
    d = clip(rng.gauss(0, rad(dpref)), -rad(dmax_deg), rad(dmax_deg))
    if rng.random() < 0.25: d = rng.choice([-1, 1]) * rng.uniform(0.4, rad(dmax_deg))
    ref = ag.psi if ag.kind == "fw" else ag.hdg
    return ag.x + L * math.cos(ref + d), ag.y + L * math.sin(ref + d)

# ---------------------------------------------------------------- CIVILIAN AIRCRAFT
def make_civilian(rng, dt):
    sub = rng.choices(["airliner", "turboprop", "light_ga"], [0.6, 0.2, 0.2])[0]
    if sub == "airliner":
        P = dict(Vmin=70, Vmax=300, a_acc=0.8, a_dec=1.2, glat=rng.uniform(0.30, 0.50), vz_max=rng.uniform(8, 15), az_max=1.0,
                 tau_om=3.0, tau_psi=8.0, tau_g=5.0, tau_pitch=2.5, zmin=30, gam_max=rad(12))
        cruise, Vc = rng.uniform(9000, 12500), rng.uniform(205, 255)
    elif sub == "turboprop":
        P = dict(Vmin=55, Vmax=170, a_acc=0.7, a_dec=1.0, glat=rng.uniform(0.30, 0.45), vz_max=rng.uniform(5, 10), az_max=1.0,
                 tau_om=2.5, tau_psi=6.0, tau_g=4.0, tau_pitch=2.0, zmin=30, gam_max=rad(12))
        cruise, Vc = rng.uniform(3000, 7000), rng.uniform(105, 150)
    else:
        P = dict(Vmin=28, Vmax=100, a_acc=0.8, a_dec=1.0, glat=rng.uniform(0.35, 0.55), vz_max=rng.uniform(2, 5), az_max=0.8,
                 tau_om=1.5, tau_psi=4.0, tau_g=3.0, tau_pitch=1.5, zmin=20, gam_max=rad(10))
        cruise, Vc = rng.uniform(300, 3500), rng.uniform(45, 75)
    ag = Agent("fw", rng, dt, **P); ag.acting = "CIVILIAN_AIRCRAFT"
    set_air_wind(ag, rng, 30 if sub == "airliner" else 15)
    prof = rng.choices(["enroute", "climb", "descent", "hold"], [0.5, 0.15, 0.2, 0.15] if sub != "light_ga" else [0.5, 0.15, 0.1, 0.25])[0]
    psi, mode = place(ag, rng, 15e3, min(140e3, horizon(cruise) * 0.9))
    z0 = cruise if prof in ("enroute", "descent", "hold") else rng.uniform(600, 2500) * (1 if sub == "airliner" else 0.5)
    if prof == "hold": z0 = cruise * rng.uniform(0.3, 0.6)
    fw_init(ag, psi, Vc if prof != "climb" else Vc * 0.6, z0)

    def legs():
        while True:
            ag.phase = "cruise"; L = rng.uniform(25e3, 110e3) if sub != "light_ga" else rng.uniform(5e3, 30e3)
            tx, ty = next_wp(ag, rng, L, 70)
            zt = cruise
            if sub == "airliner" and rng.random() < 0.12:
                zt = cruise + rng.choice([-1, 1]) * rng.uniform(300, 1200); ag.phase = "step_change"
            yield from fw_goto(ag, tx, ty, Vc, zt)
    def mission():
        if prof == "climb":
            ag.phase = "climb"; yield from fw_alt(ag, cruise, Vc, tol=200)
        elif prof == "descent":
            ag.phase = "descent"
            yield from fw_alt(ag, cruise * 0.35, Vc * 0.8, tol=150)
            yield from fw_alt(ag, cruise * 0.15, Vc * 0.55, tol=100)
            ag.phase = "approach"; ag.vz_max = 4.0
            yield from fw_alt(ag, 60.0, Vc * 0.40, tol=20)
            ag.phase = "final"; yield from fw_leg(ag, ag.psi, 1e9, Vc * 0.38, 30)
        elif prof == "hold":
            ag.phase = "holding"
            yield from fw_racetrack(ag, ag.psi, Vc * 0.6 * 60.0, Vc * 0.6, ag.z, 1e9, right=rng.random() < 0.5)
        if sub == "light_ga" and rng.random() < 0.5:
            while True:
                ag.phase = "orbit"; c = (ag.x + rng.uniform(-3e3, 3e3), ag.y + rng.uniform(-3e3, 3e3))
                yield from fw_orbit(ag, c[0], c[1], rng.uniform(800, 3000), rng.random() < 0.5, Vc, cruise, rng.uniform(120, 500))
                ag.phase = "cruise"; tx, ty = next_wp(ag, rng, rng.uniform(5e3, 20e3), 80); yield from fw_goto(ag, tx, ty, Vc, cruise)
        yield from legs()
    return ag, mission(), dict(subtype=sub, profile=prof, geometry=mode)

# ---------------------------------------------------------------- FIGHTER AIRCRAFT
def make_fighter(rng, dt, p_disguise=0.25):
    disguised = rng.random() < p_disguise
    sub = "high_speed" if disguised else rng.choices(["cap", "high_speed", "bfm", "low_level", "climb_cycles", "transit"], [0.2, 0.2, 0.2, 0.15, 0.15, 0.1])[0]
    P = dict(Vmin=110, Vmax=900, a_acc=6.0, a_dec=8.0, glat=3.0, vz_max=180, az_max=6 * G, tau_om=0.9, tau_psi=2.0, tau_a=1.5,
             tau_g=1.5, tau_pitch=0.9, tau_v=6.0, zmin=15, gam_max=rad(65), drag_k=0.07)
    ag = Agent("fw", rng, dt, **P); ag.acting = "FIGHTER_AIRCRAFT"; set_air_wind(ag, rng, 35)
    zc = rng.uniform(7000, 12000) if sub != "low_level" and sub != "climb_cycles" else rng.uniform(80, 300)
    Vc = rng.uniform(200, 260) if sub in ("cap", "transit") else rng.uniform(230, 300)
    psi, mode = place(ag, rng, 15e3, min(140e3, horizon(zc) * 0.9)); fw_init(ag, psi, Vc, zc)
    def hard_turn(n, dpsi, Vtarget):
        ag.glat = n; ag.psic = wrap(ag.psi + dpsi); ag.Vc = Vtarget; ag.phase = "hard_turn"; t = 0.0
        while abs(wrap(ag.psic - ag.psi)) > 0.12 and t < 40: yield; t += ag.dt
        ag.glat = 3.0
    def mission():
        if sub == "cap":
            ag.phase = "cap_racetrack"; Lr = rng.uniform(40e3, 70e3)
            for k in range(1000):
                yield from fw_racetrack(ag, ag.psi, Lr, Vc, zc, rng.uniform(300, 900), right=rng.random() < 0.5)
                ag.phase = "sprint"; ag.glat = 3; yield from fw_leg(ag, ag.psi, rng.uniform(20, 50), rng.uniform(380, 520), zc)
        elif sub == "high_speed":
            if disguised:
                ag.acting = "CIVILIAN_AIRCRAFT"; ag.phase = "mimic_civilian"; ag.glat = 0.4; ag.a_acc = 1.0; ag.drag_k = 0.0
                ag.Vc = rng.uniform(210, 245); ag.zc = rng.uniform(9000, 11000); ag.vz_max = 12
                ag.V = ag.Vc; ag.z = ag.zc
                T = 0.0; Tm = rng.uniform(200, 450)
                while T < Tm:
                    tx, ty = next_wp(ag, rng, rng.uniform(20e3, 50e3), 40); yield from fw_goto(ag, tx, ty, ag.Vc, ag.zc, Tmax=Tm - T); T += 60
                    T = ag.t
                ag.acting = "FIGHTER_AIRCRAFT"; ag.glat = 3.0; ag.a_acc = 6.0; ag.drag_k = 0.07; ag.vz_max = 180
                yield from hard_turn(rng.uniform(4, 7), rng.choice([-1, 1]) * rng.uniform(0.5, 1.5), 450)
            ag.phase = "high_speed_run"
            while True:
                tx, ty = next_wp(ag, rng, rng.uniform(60e3, 120e3), 50)
                yield from fw_goto(ag, tx, ty, rng.uniform(450, 750), rng.uniform(5000, 12000), Tmax=rng.uniform(90, 200))
                ag.phase = "hard_turn"; yield from hard_turn(rng.uniform(4, 7), rng.choice([-1, 1]) * rng.uniform(0.6, 2.2), 380)
                ag.phase = "high_speed_run"
        elif sub == "bfm":
            while True:
                ag.phase = "hard_turn"; yield from hard_turn(rng.uniform(4, 8.5), rng.choice([-1, 1]) * rng.uniform(0.8, 3.0), rng.uniform(220, 320))
                r = rng.random()
                if r < 0.3:
                    ag.phase = "zoom_climb"; ag.zc = min(ag.z + rng.uniform(1500, 4500), 14000); ag.Vc = 400; ag.gam_max = rad(65); t = 0.0
                    while ag.zc - ag.z > 150 and t < 25: yield; t += ag.dt
                elif r < 0.55:
                    ag.phase = "dive"; ag.zc = max(ag.z - rng.uniform(1500, 4500), 1500); ag.Vc = 450; t = 0.0
                    while ag.z - ag.zc > 150 and t < 25: yield; t += ag.dt
                ag.phase = "extend"; yield from fw_leg(ag, ag.psi, rng.uniform(4, 15), rng.uniform(250, 380), ag.zc)
        elif sub in ("low_level", "climb_cycles"):
            ag.phase = "low_level_cruise"; base = zc
            while True:
                tx, ty = next_wp(ag, rng, rng.uniform(8e3, 22e3), 45, 15)
                zt = clip(base + rng.uniform(-60, 120), 60, 400)
                yield from fw_goto(ag, tx, ty, rng.uniform(230, 290), zt)
                if rng.random() < 0.35:
                    ag.phase = "jink"; yield from hard_turn(rng.uniform(2.5, 4.0), rng.choice([-1, 1]) * rng.uniform(0.3, 0.7), 260); ag.phase = "low_level_cruise"
                if sub == "climb_cycles" and rng.random() < 0.5:
                    ag.phase = "climb_cycle"; ag.zc = ag.z + rng.uniform(1500, 2800); ag.Vc = 300; ag.gam_max = rad(45); t = 0.0
                    while ag.zc - ag.z > 150 and t < 25: yield; t += ag.dt
                    ag.phase = "descent_cycle"; ag.zc = rng.uniform(300, 600); ag.Vc = 330; t = 0.0
                    while ag.z - ag.zc > 100 and t < 30: yield; t += ag.dt
                    ag.phase = "recovery_turn"; yield from hard_turn(rng.uniform(4, 6), rng.choice([-1, 1]) * rng.uniform(1.5, 3.0), 300); ag.zc = base
                    ag.phase = "low_level_cruise"
        else:
            ag.phase = "transit"
            while True:
                tx, ty = next_wp(ag, rng, rng.uniform(30e3, 90e3), 60); yield from fw_goto(ag, tx, ty, Vc, zc)
    return ag, mission(), dict(subtype=sub + ("_disguised" if disguised else ""), geometry=mode, disguised=disguised, mimic="CIVILIAN_AIRCRAFT" if disguised else "")

# ---------------------------------------------------------------- HELICOPTER
def make_helicopter(rng, dt):
    sub = rng.choices(["transit", "observation", "search", "shuttle", "approach_land"], [0.3, 0.25, 0.2, 0.1, 0.15])[0]
    cruiseV = rng.uniform(40, 70)
    P = dict(a_max=rng.uniform(2.0, 3.4), tau_v=2.5, tau_a=1.0, vz_max=rng.uniform(4, 8), az_max=2.0, tau_vz=1.5, zmin=0.0)
    ag = Agent("vec", rng, dt, **P); ag.acting = "HELICOPTER"
    z0 = rng.uniform(40, 1200) if sub != "approach_land" else rng.uniform(150, 500)
    rmax = min(60e3, horizon(z0) * 0.9) if sub != "approach_land" else min(20e3, horizon(60) * 0.9)
    psi, mode = place(ag, rng, 3e3, rmax)
    vec_init(ag, psi, cruiseV if sub in ("transit", "approach_land", "search") else 0.0, z0)
    def stop_and_hover(T, amp=None):
        ag.phase = "decel"; ag.vcx = ag.vcy = 0.0
        while math.hypot(ag.vx, ag.vy) > 0.6: yield
        ag.phase = "hover"; ag.zc = ag.z; yield from vec_hover(ag, T, amp)
    def mission():
        if sub == "transit":
            while True:
                ag.phase = "transit"; tx, ty = next_wp(ag, rng, rng.uniform(3e3, 12e3), 60)
                zt = clip(ag.z + rng.uniform(-150, 150), 30, 1500)
                yield from vec_goto(ag, tx, ty, cruiseV, zt, tol=250, stop=False)
                if rng.random() < 0.2: yield from stop_and_hover(rng.uniform(15, 90))
        elif sub == "observation":
            yield from stop_and_hover(rng.uniform(40, 200))
            while True:
                ag.phase = "reposition"; tx, ty = next_wp(ag, rng, rng.uniform(300, 2500), 180, 60)
                yield from vec_goto(ag, tx, ty, rng.uniform(8, 35), clip(ag.z + rng.uniform(-40, 60), 30, 800), tol=15, stop=True)
                yield from stop_and_hover(rng.uniform(30, 240))
        elif sub == "search":
            while True:
                ag.phase = "search_orbit"; R = rng.uniform(300, 1500); c = (ag.x + R * math.cos(ag.hdg + 1.57), ag.y + R * math.sin(ag.hdg + 1.57))
                ccw = rng.random() < 0.5; yield from vec_orbit(ag, c[0], c[1], R, ccw, rng.uniform(22, 38), ag.z, rng.uniform(80, 240))
                ag.phase = "transit"; tx, ty = next_wp(ag, rng, rng.uniform(2e3, 6e3), 60); yield from vec_goto(ag, tx, ty, cruiseV, ag.z, tol=200, stop=False)
                if rng.random() < 0.4: yield from stop_and_hover(rng.uniform(20, 120))
        elif sub == "shuttle":
            while True:
                ag.phase = "hover_taxi"; hx, hy = next_wp(ag, rng, rng.uniform(80, 500), 180, 70)
                yield from vec_goto(ag, hx, hy, rng.uniform(3, 12), ag.z + rng.uniform(-10, 10), tol=3, stop=True)
                yield from stop_and_hover(rng.uniform(10, 50))
        else:
            ag.phase = "approach"; tx, ty = next_wp(ag, rng, rng.uniform(2e3, 5e3), 20, 10)
            yield from vec_goto(ag, tx, ty, cruiseV * 0.7, ag.z * 0.5, tol=10, stop=True)
            yield from stop_and_hover(rng.uniform(5, 25), amp=0.2)
            ag.phase = "vertical_descent"; yield from vec_vertical(ag, 4.0, rng.uniform(1.2, 3.0))
            ag.phase = "landed_hover"; ag.zc = 4.0; yield from vec_hover(ag, 20, amp=0.1, alt_amp=0.1)
            ag.done = True
    return ag, mission(), dict(subtype=sub, geometry=mode)

# ---------------------------------------------------------------- DRONE
def make_drone(rng, dt, p_disguise=0.25):
    disguised_req = rng.random() < p_disguise
    sub = "loitering" if disguised_req else rng.choices(["multirotor", "fixed_wing", "male_hale", "loitering"], [0.4, 0.25, 0.15, 0.2])[0]
    disguised = False; mimic = ""
    if sub == "multirotor":
        P = dict(a_max=rng.uniform(3.0, 7.0), tau_v=1.2, tau_a=0.6, vz_max=rng.uniform(3, 6), az_max=3.0, tau_vz=0.8, zmin=0.0)
        ag = Agent("vec", rng, dt, **P); z0 = rng.uniform(20, 450)
        psi, mode = place(ag, rng, 1e3, min(15e3, horizon(z0) * 0.9)); vec_init(ag, psi, 0.0 if rng.random() < 0.5 else rng.uniform(3, 12), z0)
    elif sub == "fixed_wing":
        P = dict(Vmin=16, Vmax=60, a_acc=1.5, a_dec=1.5, glat=rng.uniform(0.35, 0.6), vz_max=rng.uniform(3, 5), az_max=2.0, tau_om=1.0,
                 tau_psi=2.5, tau_a=1.0, tau_v=4.0, tau_g=3.0, tau_pitch=1.2, zmin=10, gam_max=rad(15))
        ag = Agent("fw", rng, dt, **P); set_air_wind(ag, rng, 12); z0 = rng.uniform(100, 2500); Vc = rng.uniform(22, 42)
        psi, mode = place(ag, rng, 3e3, min(40e3, horizon(z0) * 0.9)); fw_init(ag, psi, Vc, z0)
    elif sub == "male_hale":
        P = dict(Vmin=40, Vmax=170, a_acc=0.8, a_dec=1.0, glat=rng.uniform(0.15, 0.3), vz_max=rng.uniform(3, 8), az_max=1.0, tau_om=2.5,
                 tau_psi=6.0, tau_g=4.0, tau_pitch=2.0, zmin=50, gam_max=rad(10))
        ag = Agent("fw", rng, dt, **P); set_air_wind(ag, rng, 25); z0 = rng.uniform(4000, 15000); Vc = rng.uniform(55, 105)
        psi, mode = place(ag, rng, 10e3, min(150e3, horizon(z0) * 0.9)); fw_init(ag, psi, Vc, z0)
    else:
        P = dict(Vmin=30, Vmax=200, a_acc=3.0, a_dec=3.0, glat=rng.uniform(1.5, 3.5), vz_max=30, az_max=3 * G, tau_om=0.8, tau_psi=2.0,
                 tau_a=1.0, tau_v=4.0, tau_g=2.0, tau_pitch=1.0, zmin=3, gam_max=rad(65))
        ag = Agent("fw", rng, dt, **P); set_air_wind(ag, rng, 12); z0 = rng.uniform(100, 1500); Vc = rng.uniform(45, 100)
        psi, mode = place(ag, rng, 3e3, min(30e3, horizon(z0) * 0.9)); fw_init(ag, psi, Vc, z0)
        disguised = disguised_req; mimic = "HELICOPTER" if disguised else ""
    ag.acting = "DRONE"
    def mission():
        if sub == "multirotor":
            while True:
                r = rng.random()
                if r < 0.3:
                    ag.phase = "decel"; ag.vcx = ag.vcy = 0.0
                    while math.hypot(ag.vx, ag.vy) > 0.4: yield
                    ag.phase = "hover"; ag.zc = ag.z; yield from vec_hover(ag, rng.uniform(10, 90), amp=rng.uniform(0.2, 0.8), alt_amp=rng.uniform(0.1, 0.4))
                elif r < 0.55:
                    ag.phase = "waypoint_stop_go"; hx, hy = next_wp(ag, rng, rng.uniform(50, 500), 180, 60)
                    yield from vec_goto(ag, hx, hy, rng.uniform(4, 15), clip(ag.z + rng.uniform(-30, 40), 15, 500), tol=3, stop=True)
                    ag.phase = "hover"; ag.zc = ag.z; yield from vec_hover(ag, rng.uniform(5, 40), amp=rng.uniform(0.2, 0.7))
                elif r < 0.75:
                    ag.phase = "survey"; sp = rng.uniform(4, 8); Lleg = rng.uniform(150, 500); gap = rng.uniform(25, 60); hd = rng.uniform(0, 6.28)
                    ex, ey = math.cos(hd), math.sin(hd)
                    for k in range(rng.randint(3, 6)):
                        s = 1 if k % 2 == 0 else -1
                        yield from vec_goto(ag, ag.x + s * ex * Lleg, ag.y + s * ey * Lleg, sp, ag.z, tol=12, stop=False)
                        yield from vec_goto(ag, ag.x - ey * gap, ag.y + ex * gap, sp * 0.7, ag.z, tol=8, stop=False)
                elif r < 0.9:
                    ag.phase = "orbit_poi"; R = rng.uniform(30, 150); sp = rng.uniform(3, 8); hd = ag.hdg
                    c = (ag.x + R * math.cos(hd + 1.57), ag.y + R * math.sin(hd + 1.57)); yield from vec_orbit(ag, c[0], c[1], R, rng.random() < .5, sp, ag.z, rng.uniform(40, 150))
                elif r < 0.96:
                    ag.phase = "vertical"; yield from vec_vertical(ag, clip(ag.z + rng.choice([-1, 1]) * rng.uniform(30, 150), 15, 500), rng.uniform(2, 5))
                else:
                    ag.phase = "fast_dash"; ag.a_max = 9.0; hx, hy = next_wp(ag, rng, rng.uniform(300, 900), 40, 20)
                    yield from vec_goto(ag, hx, hy, rng.uniform(20, 32), ag.z, tol=30, stop=False); ag.a_max = P["a_max"]
        elif sub == "fixed_wing":
            while True:
                r = rng.random()
                if r < 0.35:
                    ag.phase = "loiter"; R = rng.uniform(150, 500); c = (ag.x + R * math.cos(ag.psi + 1.57), ag.y + R * math.sin(ag.psi + 1.57))
                    yield from fw_orbit(ag, c[0], c[1], R, rng.random() < .5, Vc, ag.z, rng.uniform(60, 300))
                elif r < 0.7:
                    ag.phase = "waypoint"; tx, ty = next_wp(ag, rng, rng.uniform(1e3, 6e3), 100, 45)
                    yield from fw_goto(ag, tx, ty, Vc, clip(ag.z + rng.uniform(-150, 200), 60, 3000))
                else:
                    ag.phase = "survey"; Lleg = rng.uniform(1.5e3, 4e3); gap = rng.uniform(250, 500); ex, ey = math.cos(ag.psi), math.sin(ag.psi); sg = 1
                    for k in range(rng.randint(3, 5)):
                        yield from fw_goto(ag, ag.x + ex * Lleg * sg, ag.y + ey * Lleg * sg, Vc, ag.z); 
                        nx, ny = -ey, ex
                        yield from fw_goto(ag, ag.x + nx * gap, ag.y + ny * gap, Vc, ag.z, tol=150); sg = -sg
        elif sub == "male_hale":
            while True:
                r = rng.random()
                if r < 0.4:
                    ag.phase = "racetrack"; yield from fw_racetrack(ag, ag.psi, rng.uniform(10e3, 40e3), Vc, ag.z, rng.uniform(300, 900), right=rng.random() < .5)
                elif r < 0.7:
                    ag.phase = "orbit"; R = rng.uniform(3e3, 8e3); c = (ag.x + R * math.cos(ag.psi + 1.57), ag.y + R * math.sin(ag.psi + 1.57))
                    yield from fw_orbit(ag, c[0], c[1], R, rng.random() < .5, Vc, ag.z, rng.uniform(300, 900))
                else:
                    ag.phase = "transit"; tx, ty = next_wp(ag, rng, rng.uniform(20e3, 60e3), 50)
                    yield from fw_goto(ag, tx, ty, Vc, clip(ag.z + rng.uniform(-1000, 1000), 3000, 15000))
        else:
            if disguised:
                ag.acting = "HELICOPTER"; ag.phase = "mimic_helicopter"; vm = rng.uniform(38, 55); ag.glat = 0.35
                ag.Vc = ag.V = vm; ag.zc = ag.z = rng.uniform(100, 600); T = rng.uniform(150, 400)
                while ag.t < T:
                    if rng.random() < .5: tx, ty = next_wp(ag, rng, rng.uniform(1.5e3, 4e3), 60); yield from fw_goto(ag, tx, ty, vm, ag.z, Tmax=T - ag.t)
                    else: c = (ag.x + 400 * math.cos(ag.psi + 1.57), ag.y + 400 * math.sin(ag.psi + 1.57)); yield from fw_orbit(ag, c[0], c[1], 400, rng.random() < .5, vm, ag.z, min(120, max(T - ag.t, 5)))
                ag.acting = "DRONE"; ag.glat = P["glat"]
            for _ in range(rng.randint(0, 2)):
                ag.phase = "ingress"; tx, ty = next_wp(ag, rng, rng.uniform(2e3, 6e3), 70); yield from fw_goto(ag, tx, ty, rng.uniform(60, 110), clip(ag.z, 150, 1500))
            ag.phase = "loiter"; R = rng.uniform(400, 1000); c = (ag.x + R * math.cos(ag.psi + 1.57), ag.y + R * math.sin(ag.psi + 1.57))
            if not disguised and rng.random() < 0.6: yield from fw_orbit(ag, c[0], c[1], R, rng.random() < .5, rng.uniform(45, 80), ag.z, rng.uniform(40, 150))
            ag.phase = "descent"; ag.vz_max = 15.0; ag.Vc = rng.uniform(55, 90)
            yield from fw_alt(ag, rng.uniform(60, 200), ag.Vc, tol=30)
            while True:
                ag.phase = "low_cruise"; tx, ty = next_wp(ag, rng, rng.uniform(1.5e3, 5e3), 70, 30); yield from fw_goto(ag, tx, ty, ag.Vc, ag.z)
    return ag, mission(), dict(subtype=sub + ("_disguised" if disguised else ""), geometry=mode, disguised=disguised, mimic=mimic)

# ---------------------------------------------------------------- MISSILE CLASSES: PLACEHOLDER PROFILES ONLY
# Simple scripted speed / height / turn profiles (no guidance, homing or target logic). Replace with approved data later.
def make_surface_missile(rng, dt):
    sub = rng.choice(["subsonic_profile", "supersonic_profile"])
    Vc = rng.uniform(220, 300) if sub == "subsonic_profile" else rng.uniform(600, 900)
    zc = rng.uniform(8, 20)
    P = dict(Vmin=60, Vmax=1000, a_acc=10.0, a_dec=5.0, glat=rng.uniform(2, 4), vz_max=40, az_max=3 * G, tau_om=0.6, tau_psi=2.0, tau_a=0.8,
             tau_v=5.0, tau_g=1.5, tau_pitch=0.8, zmin=3, gam_max=rad(25))
    ag = Agent("fw", rng, dt, **P); ag.acting = "SURFACE_MISSILE"; set_air_wind(ag, rng, 8)
    psi, mode = place(ag, rng, 4e3, min(40e3, horizon(zc) * 0.9)); fw_init(ag, psi, Vc * 0.6, zc)
    def mission():
        ag.phase = "accelerate"; ag.Vc = Vc; ag.zc = zc
        yield from hold(ag, 12)
        while True:
            ag.phase = "level_cruise"
            yield from fw_leg(ag, ag.psi + rng.uniform(-0.25, 0.25), rng.uniform(15, 40), Vc, zc)
    return ag, mission(), dict(subtype=sub + "_PLACEHOLDER", geometry=mode)

def make_aerial_missile(rng, dt):
    sub = rng.choice(["short_range_profile", "long_range_profile"])
    z0 = rng.uniform(3000, 11000); V0 = rng.uniform(220, 300)
    P = dict(Vmin=100, Vmax=1400, glat=rng.uniform(3, 8), tau_om=0.4, tau_psi=2.0, tau_a=0.5, az_max=4 * G, tau_g=1.5, tau_pitch=0.6,
             zmin=100, gam_max=rad(40), vz_max=150)
    ag = Agent("fw", rng, dt, **P); ag.acting = "AERIAL_MISSILE"; set_air_wind(ag, rng, 5)
    psi, mode = place(ag, rng, 5e3, min(90e3, horizon(z0) * 0.9)); fw_init(ag, psi, V0, z0)
    tb = rng.uniform(2.5, 5.0) if sub == "short_range_profile" else rng.uniform(4.0, 8.0)
    ab = rng.uniform(12, 22) * G; k = 1.2 * G / 700.0 ** 2
    def mission():
        ag.phase = "boost"; ag.al_cmd = ab; ag.zc = z0
        for _ in range(int(tb / ag.dt)): yield
        ag.phase = "coast"; psi_c = ag.psi
        while True:
            ag.al_cmd = -k * math.exp(-ag.z / 8500.0) * ag.V ** 2
            ag.psic = psi_c + 0.15 * math.sin(0.05 * ag.t); ag.zc = z0 + (3000 if sub == "long_range_profile" else 0) * min(ag.t / 20.0, 1.0)
            if ag.V < 150: ag.done = True; return
            yield
    return ag, mission(), dict(subtype=sub + "_PLACEHOLDER", geometry=mode)

# ---------------------------------------------------------------- SURFACE VESSELS (skiff / boat / ship) with disguises
def make_surface(rng, dt, cls, p_disguise=0.25):
    if cls == "SKIFF":
        mimic_cls = "BOAT"; sub = rng.choices(["transit", "zigzag", "fishing", "dash", "circle"], [0.3, 0.15, 0.25, 0.15, 0.15])[0]
        P = dict(Vmin=0.0, Vmax=32, a_acc=rng.uniform(1.0, 3.0), a_dec=rng.uniform(1.5, 3.0), om_cap=rng.uniform(0.2, 0.5), tau_v=3.0, tau_a=1.5,
                 tau_om=1.2, tau_psi=2.5, zmin=0.5); z_ref = rng.uniform(1, 3); hv = [(rng.uniform(0.3, 0.8), rng.uniform(3, 6), rng.uniform(0, 6.28)), (rng.uniform(0.1, 0.3), rng.uniform(2, 4), rng.uniform(0, 6.28))]
        Vtr = (8, 20); Vlow = (0.3, 2.5)
    elif cls == "BOAT":
        mimic_cls = "SHIP"; sub = rng.choices(["transit", "patrol", "fishing", "dash", "circle"], [0.3, 0.25, 0.2, 0.1, 0.15])[0]
        P = dict(Vmin=0.0, Vmax=40, a_acc=rng.uniform(0.4, 1.0), a_dec=rng.uniform(0.5, 1.2), om_cap=rng.uniform(0.06, 0.15), tau_v=6.0, tau_a=3.0,
                 tau_om=3.0, tau_psi=6.0, zmin=0.5); z_ref = rng.uniform(3, 8); hv = [(rng.uniform(0.2, 0.6), rng.uniform(4, 7), rng.uniform(0, 6.28)), (rng.uniform(0.1, 0.3), rng.uniform(3, 5), rng.uniform(0, 6.28))]
        Vtr = (5, 18); Vlow = (0.5, 3.0)
    else:
        mimic_cls = None; sub = rng.choices(["transit", "zigzag", "drift", "racetrack", "anchored"], [0.4, 0.15, 0.15, 0.15, 0.15])[0]
        P = dict(Vmin=0.0, Vmax=22, a_acc=rng.uniform(0.04, 0.15), a_dec=rng.uniform(0.05, 0.2), om_cap=rng.uniform(0.006, 0.03), tau_v=60.0, tau_a=20.0,
                 tau_om=20.0, tau_psi=60.0, zmin=0.5); z_ref = rng.uniform(8, 35); hv = [(rng.uniform(0.05, 0.3), rng.uniform(7, 12), rng.uniform(0, 6.28))]
        Vtr = (4, 15); Vlow = (0.1, 1.0)
    disguised = bool(mimic_cls) and rng.random() < p_disguise
    ag = Agent("fw", rng, dt, planar=True, **P); ag.acting = cls; ag.z_ref = z_ref; ag.heave = hv; ag.zc = z_ref
    cur = rng.uniform(0, 0.6); d = rng.uniform(0, 6.28); ag.wind = (cur * math.cos(d), cur * math.sin(d)); ag.wind_ref = 4.0
    psi, mode = place(ag, rng, 3e3, min(60e3, horizon(z_ref) * 0.9)); V0 = rng.uniform(*Vtr)
    fw_init(ag, psi, V0, z_ref); ag.z = z_ref
    real_om = ag.om_cap
    def mission():
        if disguised:
            ag.acting = mimic_cls; ag.phase = "mimic_" + mimic_cls.lower()
            ag.om_cap = 0.015 if mimic_cls == "SHIP" else 0.1; Vm = rng.uniform(1.5, 3.0) if cls == "SKIFF" else rng.uniform(5, 9)
            ag.V = ag.Vc = Vm                      # start already at the disguise speed (no start-up transient)
            ag.a_acc = min(ag.a_acc, 0.3); T = rng.uniform(250, 600)
            while ag.t < T:
                tx, ty = next_wp(ag, rng, rng.uniform(1.5e3, 5e3), 40, 18); yield from fw_goto(ag, tx, ty, Vm, z_ref, tol=100, Tmax=T - ag.t)
            ag.acting = cls; ag.om_cap = real_om; ag.a_acc = P["a_acc"]; ag.phase = "breakout_dash"
            tx, ty = next_wp(ag, rng, rng.uniform(4e3, 9e3), 80, 30); yield from fw_goto(ag, tx, ty, rng.uniform(*(20, 30) if cls == "SKIFF" else (22, 34)), z_ref, tol=100, Tmax=150)
        while True:
            ag.phase = sub
            if sub in ("transit", "patrol"):
                tx, ty = next_wp(ag, rng, rng.uniform(2e3, 9e3) if cls != "SHIP" else rng.uniform(6e3, 25e3), 70, 25)
                yield from fw_goto(ag, tx, ty, rng.uniform(*Vtr), z_ref, tol=150)
            elif sub == "zigzag":
                base = ag.psi
                for _ in range(rng.randint(3, 8)):
                    yield from fw_leg(ag, base + rng.choice([-1, 1]) * rng.uniform(0.3, 1.0 if cls == "SKIFF" else 0.45), rng.uniform(10, 40) if cls == "SKIFF" else rng.uniform(150, 400), rng.uniform(*Vtr), z_ref)
            elif sub in ("fishing", "drift", "anchored"):
                v = rng.uniform(*Vlow) if sub == "fishing" else rng.uniform(0.05, 0.6)
                for _ in range(rng.randint(2, 5)):
                    yield from fw_leg(ag, ag.psi + rng.uniform(-1.2, 1.2), rng.uniform(40, 200) if cls != "SHIP" else rng.uniform(200, 600), v, z_ref)
            elif sub == "dash":
                tx, ty = next_wp(ag, rng, rng.uniform(3e3, 8e3), 50, 20); yield from fw_goto(ag, tx, ty, rng.uniform(22, 31) if cls == "SKIFF" else rng.uniform(20, 36), z_ref, tol=150, Tmax=200)
                yield from fw_leg(ag, ag.psi, rng.uniform(20, 60), rng.uniform(*Vlow), z_ref)
            elif sub == "circle":
                R = rng.uniform(80, 300) if cls == "SKIFF" else rng.uniform(200, 700); c = (ag.x + R * math.cos(ag.psi + 1.57), ag.y + R * math.sin(ag.psi + 1.57))
                yield from fw_orbit(ag, c[0], c[1], R, rng.random() < .5, rng.uniform(4, 12), z_ref, rng.uniform(60, 200))
            elif sub == "racetrack":
                yield from fw_racetrack(ag, ag.psi, rng.uniform(6e3, 12e3), rng.uniform(8, 14), z_ref, rng.uniform(600, 1500), right=rng.random() < .5, tol=200)
    return ag, mission(), dict(subtype=sub + ("_disguised" if disguised else ""), geometry=mode, disguised=disguised, mimic=mimic_cls if disguised else "")

FACTORIES = {"CIVILIAN_AIRCRAFT": make_civilian, "FIGHTER_AIRCRAFT": make_fighter, "HELICOPTER": make_helicopter, "DRONE": make_drone,
             "SURFACE_MISSILE": make_surface_missile, "AERIAL_MISSILE": make_aerial_missile,
             "SKIFF": lambda r, d: make_surface(r, d, "SKIFF"), "BOAT": lambda r, d: make_surface(r, d, "BOAT"), "SHIP": lambda r, d: make_surface(r, d, "SHIP")}
FINE_DT = {"FIGHTER_AIRCRAFT": 0.125, "SURFACE_MISSILE": 0.05, "AERIAL_MISSILE": 0.05, "HELICOPTER": 0.1, "DRONE": 0.1}
T_RANGE = {"CIVILIAN_AIRCRAFT": (400, 1500), "FIGHTER_AIRCRAFT": (250, 700), "HELICOPTER": (300, 1000), "DRONE": (200, 900),
           "SURFACE_MISSILE": (60, 120), "AERIAL_MISSILE": (20, 60), "SKIFF": (300, 900), "BOAT": (500, 1500), "SHIP": (600, 1800)}
MAX_RANGE = 250e3

def simulate_track(cls, seed, track_no, t0=1.76e9, min_packets=8, **kw):
    """One noise-free track sampled every 2.5 s. Returns (DataFrame, meta)."""
    for attempt in range(20):
        rng = random.Random(seed * 100 + attempt); dt = FINE_DT.get(cls, 0.25)
        ag, gen, meta = FACTORIES[cls](rng, dt)
        Tcap = rng.uniform(*T_RANGE[cls])
        if meta.get("disguised"): Tcap = max(Tcap, 1000.0 if cls in ("SKIFF", "BOAT") else 800.0)
        n_sub = int(round(DT_OUT / dt)); rows = []; k = 0
        while ag.t < Tcap - 1e-9 and not ag.done:
            try: next(gen)                      # behaviour sets commands / phase / acting class first ...
            except StopIteration: break
            if k % n_sub == 0:                  # ... then the packet is sampled, so labels are correct from packet 0
                x, y, z, v, a, h = ag.sample(); rows.append((ag.t, x, y, z, v, a, h, ag.acting, ag.phase))
            ag.step(); k += 1
        if k % n_sub == 0 and not ag.done:
            x, y, z, v, a, h = ag.sample(); rows.append((ag.t, x, y, z, v, a, h, ag.acting, ag.phase))
        # radar visibility: keep the longest contiguous stretch inside the radar horizon and max range
        vis = [(math.hypot(r[1], r[2]) < min(MAX_RANGE, horizon(r[3]))) for r in rows]
        best = (0, 0); i = 0
        while i < len(vis):
            if vis[i]:
                j = i
                while j < len(vis) and vis[j]: j += 1
                if j - i > best[1] - best[0]: best = (i, j)
                i = j
            else: i += 1
        rows = rows[best[0]:best[1]]
        if len(rows) >= min_packets: break
    else:
        raise RuntimeError(f"could not build a visible {cls} track (seed {seed})")
    n = len(rows); t_rel = np.arange(n) * DT_OUT
    P = np.array([[r[1], r[2], r[3]] for r in rows]); V = np.array([r[4] for r in rows]); A = np.array([r[5] for r in rows])
    hd = np.degrees(np.array([r[6] for r in rows])) % 360
    tt = t0 + t_rel
    df = pd.DataFrame({"packet_size": 166, "source": 7, "destination": 255, "packet_type": 163, "packet_category": 11, "radar_id": 1, "track_no": track_no,
        "pos_x": P[:, 0], "pos_y": P[:, 1], "pos_z": P[:, 2], "range_mag": np.linalg.norm(P, axis=1),
        "vel_x": V[:, 0], "vel_y": V[:, 1], "vel_z": V[:, 2], "vel_mag": np.linalg.norm(V, axis=1), "heading": hd,
        "accel_x": A[:, 0], "accel_y": A[:, 1], "accel_z": A[:, 2], "accel_mag": np.linalg.norm(A, axis=1)})
    for c in ["std_dev_x", "std_dev_y", "std_dev_z", "std_dev_vx", "std_dev_vy", "std_dev_vz", "std_dev_ax", "std_dev_ay", "std_dev_az", "cov_x_vx", "cov_y_vy", "cov_z_vz"]:
        df[c] = 0.0
    df["ntp_sec"] = np.floor(tt).astype(np.int64); df["ntp_frac"] = ((tt % 1) * 1e6).round().astype(np.int64); df["assoc_indication"] = 1
    sp = np.linalg.norm(V, axis=1); vhat = V / np.maximum(sp, 1.0)[:, None]
    a_al = (A * vhat).sum(1); a_lat = np.linalg.norm(A - a_al[:, None] * vhat, axis=1)
    df["time_s"] = t_rel; df["true_class"] = cls; df["acting_class"] = [r[7] for r in rows]; df["phase"] = [r[8] for r in rows]
    df["subtype"] = meta["subtype"]; df["geometry"] = meta.get("geometry", ""); df["disguised"] = bool(meta.get("disguised", False))
    df["lateral_g"] = a_lat / G; df["speed"] = sp
    return df, meta
