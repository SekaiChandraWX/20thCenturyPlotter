import streamlit as st
import requests
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import io
from datetime import datetime
from bs4 import BeautifulSoup

# ==================== Streamlit page ====================
st.set_page_config(page_title="20CR Global Plotter — Dewpoint/MSLP & 500/850 mb Winds", layout="wide")

# ==================== Regions (global, IDL-safe) ====================
# (lon_w, lon_e, lat_s, lat_n). Longitudes may be outside [-180,180] or span across the IDL.
REGIONS = {
    "General": {
        "Global (60°S–60°N)": [-180, 180, -60, 60],
        "Worldwide (80°S–80°N)": [-180, 180, -80, 80],
        "Continental United States": [-125, -66.5, 24.396, 49.384],
        "Europe": [-12.2, 49.4, 26.6, 74.3],
        "Middle East and South Asia": [27.9, 102.3, 1.8, 67.5],
        "East and Southeast Asia": [86.4, 160.8, -14.7, 50.9],
        "Australia and Oceania": [108.8, 191.0, -52.6, -5.8],  # crosses IDL
        "Northern Africa": [-20.5, 55.6, -4.2, 39.4],
        "Southern Africa": [2.8, 59.5, -39.5, -4.7],
        "Northern South America": [-83.5, -31.3, -24.2, 13.7],
        "Southern South America": [-86.4, -34.3, -58.4, -15.7],
        "Mainland Canada": [-128, -52, 40.6, 62.7],
        "Mexico and Central America": [-119, -56.1, 3.3, 35.7]
    },
    "Tropics / TC Basins": {
        "North Atlantic Basin": [-102.8, -7.9, 6, 57.6],
        "West Pacific Basin": [94.9, 183.5, -14.6, 56.1],        # crosses IDL
        "East Pacific Basin": [-161.4, -86.3, 3, 39],
        "Central Pacific Basin": [-188.8, -141.6, 2.4, 41.1],    # crosses IDL
        "Northern Indian Ocean Basin": [-317, -256.3, -5, 34],   # crosses IDL in 0–360 wrap
        "South Indian Ocean Basin": [32.7, 125.4, -44.8, 3.5],
        "Australian Basin": [100, 192.7, -50.2, -1.9]            # crosses IDL
    }
}

# ==================== Color maps (match your scripts) ====================
def dewpoint_colormap_and_levels():
    cols = [(152,109,77),(150,108,76),(148,107,76),(146,106,75),(144,105,75),(142,104,74),
            (140,102,74),(138,101,73),(136,100,72),(134,99,72),(132,98,71),(130,97,71),
            (128,96,70),(126,95,70),(124,94,69),(122,93,68),(120,91,68),(118,90,67),
            (116,89,67),(114,88,66),(113,87,66),(111,86,65),(109,85,64),(107,84,64),
            (105,83,63),(103,82,63),(101,80,62),(99,79,61),(97,78,61),(95,77,60),
            (93,76,60),(91,75,59),(89,74,59),(87,73,58),(85,72,57),(83,71,57),
            (81,69,56),(79,68,56),(77,67,55),(75,66,55),(73,65,54),(71,64,54),
            (69,63,53),(77,67,52),(81,71,56),(86,76,60),(90,80,65),(94,85,69),
            (99,89,73),(103,94,77),(107,98,81),(112,103,86),(116,107,90),(120,112,94),
            (125,116,98),(129,121,103),(133,125,107),(138,130,111),(142,134,115),
            (146,139,119),(151,143,124),(155,148,128),(159,152,132),(164,157,137),
            (168,161,141),(173,166,145),(189,179,156),(189,179,156),(188,184,161),
            (193,188,165),(201,197,173),(201,197,173),(210,206,182),(223,220,194),
            (227,224,198),(231,229,202),(235,233,207),(240,238,211),(244,242,215),
            (230,245,230),(215,240,215),(200,234,200),(185,229,185),(170,223,170),
            (155,218,155),(140,213,140),(125,207,125),(110,202,110),(95,196,95),
            (80,191,80),(65,186,65),(48,174,48),(44,163,44),(39,153,39),(35,142,35),
            (30,131,30),(26,121,26),(21,110,21),(17,99,17),(12,89,12),(8,78,8),
            (97,163,175),(88,150,160),(80,137,146),(71,123,131),(62,110,116),
            (54,97,102),(45,84,87),(36,70,72),(28,57,58),(19,44,43),(102,102,154),
            (96,94,148),(89,86,142),(83,78,136),(77,70,130),(70,62,124),(64,54,118),
            (58,46,112),(51,38,106),(45,30,100),(114,64,113),(120,69,115),(125,75,117),
            (131,80,118),(136,86,120),(142,91,122),(147,97,124),(153,102,125),
            (158,108,127),(164,113,129)]
    cols = [(r/255, g/255, b/255) for r,g,b in cols]
    levels = np.linspace(-40, 90, len(cols) + 1)
    cmap = mcolors.ListedColormap(cols)
    norm = mcolors.BoundaryNorm(levels, len(cols))
    return cmap, norm, levels

def cmap_500_wind():
    pw500 = [
        (230,244,255),(219,240,254),(209,235,254),(198,231,253),(188,227,253),(177,223,252),
        (167,219,252),(156,214,251),(146,210,251),(135,206,250),(132,194,246),(129,183,241),
        (126,171,237),(123,160,232),(121,148,228),(118,136,223),(115,125,219),(112,113,214),
        (109,102,210),(106,90,205),(118,96,207),(131,102,208),(143,108,210),(156,114,211),
        (168,120,213),(180,126,214),(193,132,216),(205,138,217),(218,144,219),(230,150,220),
        (227,144,217),(224,138,214),(221,132,211),(218,126,208),(215,120,205),(212,114,202),
        (209,108,199),(206,102,196),(203,96,193),(200,90,190),(196,83,186),(192,76,182),
        (188,69,178),(184,62,174),(180,55,170),(176,48,166),(172,41,162),(168,34,158),
        (164,27,154),(160,20,150),(164,16,128),(168,14,117),(172,14,117),(176,12,106),
        (180,8,95),(184,6,73),(188,4,62),(192,2,51),(200,0,40),(200,0,40),(202,4,42),
        (204,8,44),(208,12,44),(210,20,50),(212,24,52),(212,24,52),(214,28,54),(218,36,58),
        (220,40,60),(222,44,62),(224,48,64),(226,52,66),(228,56,68),(230,60,70),(232,64,72),
        (234,68,74),(236,72,76),(238,76,78),(240,80,80),(241,96,82),(242,112,84),(243,128,86),
        (244,144,88),(245,160,90),(246,176,92),(247,192,94),(248,208,96),(249,224,98),(250,240,100),
        (247,235,97),(244,230,94),(241,225,91),(238,220,88),(235,215,85),(232,210,82),(229,205,79),
        (226,200,76),(223,195,73),(220,190,70),(217,185,67),(214,180,64),(211,175,61),(208,170,58),
        (205,165,55),(202,160,52),(199,155,49),(196,150,46),(193,145,43),(190,140,40),(187,135,37),
        (184,130,34),(181,125,31),(178,120,28),(175,115,25),(172,110,22),(169,105,19),(166,100,16),
        (163,95,13),(160,90,10),(160,90,10)
    ]
    return mcolors.ListedColormap([(r/255, g/255, b/255) for r,g,b in pw500])

def cmap_850_wind():
    # 850 palette close to your 20–80 kt style
    pw850 = [
        (240,248,255),(219,240,254),(198,231,253),(177,223,252),(156,214,251),(135,206,250),
        (129,183,241),(123,160,232),(118,136,223),(112,113,214),(106,90,205),(131,102,208),
        (156,114,211),(180,126,214),(205,138,217),(230,150,220),(224,138,214),(218,126,208),
        (212,114,202),(206,102,196),(200,90,190),(192,76,182),(184,62,174),(176,48,166),
        (168,34,158),(160,20,150),(168,16,128),(176,12,106),(184,4,62),(200,0,40),
        (200,0,40),(208,16,48),(212,24,52),(216,32,56),(220,40,60),(224,48,64),
        (228,56,68),(232,64,72),(236,72,76),(240,80,80),(242,112,84),(244,144,88),
        (246,176,92),(248,208,96),(250,240,100),(244,230,94),(238,220,88),(232,210,82),
        (226,200,76),(220,190,70),(214,180,64),(208,170,58),(202,160,52),(196,150,46),
        (190,140,40),(184,130,34),(178,120,28),(172,110,22),(166,100,16),(160,90,10)
    ]
    return mcolors.ListedColormap([(r/255, g/255, b/255) for r,g,b in pw850])

# ==================== Dateline-safe recentering & subsetting ====================
def mod360(x): return (np.asarray(x) % 360.0 + 360.0) % 360.0

def shortest_arc_mid(lw, le):
    w = mod360(lw); e = mod360(le)
    d = (e - w) % 360.0
    if d <= 180.0:
        center = (w + d / 2.0) % 360.0
        w_u, e_u = w, w + d
    else:
        d2 = 360.0 - d
        center = (e + d2 / 2.0) % 360.0
        w_u, e_u = w, w + 360.0 - d2
    return float(center), float(w_u), float(e_u)

def build_projection_and_extent(lon_w, lon_e, lat_s, lat_n):
    s, n = (lat_s, lat_n) if lat_s <= lat_n else (lat_n, lat_s)
    center, w_u, e_u = shortest_arc_mid(lon_w, lon_e)
    def to_center_frame(lon):
        return ((lon - center + 180.0) % 360.0) - 180.0
    w_c = to_center_frame(w_u); e_c = to_center_frame(e_u)
    proj = ccrs.PlateCarree(central_longitude=center)
    extent_crs = ccrs.PlateCarree(central_longitude=center)
    extent = [w_c, e_c, s, n]
    return proj, extent_crs, extent, center

def subset_lon_lat(lon_360, lat, data3d, lon_w, lon_e, lat_s, lat_n):
    """lon_360: strictly increasing 0..360; data3d: [time, lat, lon]."""
    lon_pm180 = ((lon_360 + 180.0) % 360.0) - 180.0
    s, n = (lat_s, lat_n) if lat_s <= lat_n else (lat_n, lat_s)
    center, w_u, e_u = shortest_arc_mid(lon_w, lon_e)
    cmp_axis = mod360(lon_pm180)
    cmp_unwrapped = np.where(cmp_axis < w_u, cmp_axis + 360.0, cmp_axis)
    idx = np.where((cmp_unwrapped >= w_u) & (cmp_unwrapped <= e_u))[0]
    if idx.size == 0:
        idx = np.arange(lon_pm180.size)
    idx = idx[np.argsort(cmp_unwrapped[idx])]
    lon_sel = lon_pm180[idx]

    # latitude selection robust to descending arrays
    if lat[0] < lat[-1]:
        lat_mask = (lat >= s) & (lat <= n)
    else:
        lat_mask = (lat <= n) & (lat >= s)
    lat_sel = lat[lat_mask]
    sub = data3d[:, lat_mask, :][:, :, idx]
    return lon_sel, lat_sel, sub

def ensure_increasing_axes(lon1d, lat1d, *fields):
    lon_up, lat_up, out = lon1d, lat1d, list(fields)
    if lon_up[0] > lon_up[-1]:
        lon_up = lon_up[::-1]
        out = [np.ascontiguousarray(f[..., ::-1]) for f in out]
    if lat_up[0] > lat_up[-1]:
        lat_up = lat_up[::-1]
        out = [np.ascontiguousarray(f[:, ::-1, :]) for f in out]
    return (lon_up, lat_up, *out)

def to_center_frame_vec(lon_pm180, center_deg):
    return ((lon_pm180 - center_deg + 180.0) % 360.0) - 180.0

# ==================== UI helpers ====================
def ordinal_suffix(d):
    if 10 <= d % 100 <= 20: return "th"
    return {1:"st",2:"nd",3:"rd"}.get(d % 10,"th")

def date_title_str(year, month, day, hour):
    dt = datetime(year, month, day, hour)
    d = dt.day
    return f"{dt.strftime('%B')} {d}{ordinal_suffix(d)}, {dt.year} at {dt:%H}:00 UTC"

def auto_plot_params(extent, nx, ny):
    w, e, s, n = extent
    lon_span = e - w
    lat_span = abs(n - s)
    span = max(lon_span, lat_span)
    if span >= 120:
        desired_x = 28; barb_len = 5; barb_min_stride = 9
        mslp_lw = 0.95; coast_lw = 0.9; border_lw = 0.75; state_lw = 0.5; cint = 3
    elif span >= 60:
        desired_x = 45; barb_len = 6; barb_min_stride = 8
        mslp_lw = 1.0; coast_lw = 1.0; border_lw = 0.8; state_lw = 0.6; cint = 2
    elif span >= 30:
        desired_x = 65; barb_len = 6; barb_min_stride = 7
        mslp_lw = 1.05; coast_lw = 1.0; border_lw = 0.8; state_lw = 0.6; cint = 3
    else:
        desired_x = 85; barb_len = 6; barb_min_stride = 6
        mslp_lw = 1.1; coast_lw = 1.0; border_lw = 0.8; state_lw = 0.6; cint = 4
    stride_x = max(1, min(14, max(nx // desired_x, barb_min_stride)))
    stride_y = max(1, min(14, max(ny // int(desired_x / 1.6), barb_min_stride)))
    return {
        'stride_y': stride_y, 'stride_x': stride_x, 'barb_len': barb_len,
        'mslp_lw': mslp_lw, 'coast_lw': coast_lw, 'border_lw': border_lw, 'state_lw': state_lw, 'cint': cint
    }

def plot_base_map(ax, params):
    ax.add_feature(cfeature.COASTLINE, linewidth=params['coast_lw'])
    ax.add_feature(cfeature.BORDERS, linewidth=params['border_lw'])
    try:
        ax.add_feature(cfeature.STATES, linewidth=params['state_lw'])
    except Exception:
        pass

# ==================== PSL helpers (HTML→NetCDF) ====================
def build_psl_url_level(var_name: str, level_str: str, year: int, month: int, day: int, hour: int) -> str:
    return (
        f"https://psl.noaa.gov/cgi-bin/data/composites/comp.20thc.hour.pl?var={var_name}"
        f"&level={level_str}&version=3&iy%5B1%5D={year}&im%5B1%5D={month}&id%5B1%5D={day}&ih%5B1%5D={hour:02d}"
        "&monr1=1&dayr1=1&hour1=0&monr2=1&dayr2=1&hour2=0&proj=USA&xlat1=&xlat2=&xlon1=&xlon2="
        "&custproj=Cylindrical+Equidistant&level1=1000mb&level2=10mb&state=0&Submit=Create+Plot"
    )

def build_psl_url_sfc(var_name: str, year: int, month: int, day: int, hour: int) -> str:
    # for dewpoint/MSLP/10m winds (single-levels)
    return (
        f"https://psl.noaa.gov/cgi-bin/data/composites/comp.20thc.hour.pl?var={var_name}"
        f"&level=1000mb&version=3&iy%5B1%5D={year}&im%5B1%5D={month}&id%5B1%5D={day}&ih%5B1%5D={hour:02d}"
        "&monr1=1&dayr1=1&hour1=0&monr2=1&dayr2=1&hour2=0&proj=USA&xlat1=&xlat2=&xlon1=&xlon2="
        "&custproj=Cylindrical+Equidistant&level1=1000mb&level2=10mb&state=0&Submit=Create+Plot"
    )

def fetch_netcdf_from_psl(url: str, file_prefix: str, rename_to: str) -> str:
    r = requests.get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    netcdf_url = None
    for a in soup.find_all("a", href=True):
        if a["href"].endswith(".nc") and file_prefix in a["href"]:
            netcdf_url = "https://psl.noaa.gov" + a["href"]
            break
    if not netcdf_url:
        for a in soup.find_all("a", href=True):
            if "Get a copy of the netcdf data file" in a.text:
                netcdf_url = "https://psl.noaa.gov" + a["href"]
                break
    if not netcdf_url:
        raise RuntimeError("NetCDF file link not found on PSL page.")

    os.makedirs("netcdf_files", exist_ok=True)
    local_path = os.path.join("netcdf_files", rename_to)
    with open(local_path, "wb") as f:
        f.write(requests.get(netcdf_url).content)
    return local_path

def download_level_bundle(level_mb: int, year: int, month: int, day: int, hour: int, tag: str):
    level_str = f"{int(level_mb)}mb"
    base = f"{year:04d}{month:02d}{day:02d}{hour:02d}"
    u_file = fetch_netcdf_from_psl(
        build_psl_url_level("Vector+Wind", level_str, year, month, day, hour),
        "comphour", f"u_{level_mb}_{base}_{tag}.nc"
    )
    v_file = fetch_netcdf_from_psl(
        build_psl_url_level("Vector+Wind", level_str, year, month, day, hour),
        "vcomphour", f"v_{level_mb}_{base}_{tag}.nc"
    )
    h_file = fetch_netcdf_from_psl(
        build_psl_url_level("Geopotential+Height", level_str, year, month, day, hour),
        "comphour", f"hgt_{level_mb}_{base}_{tag}.nc"
    )
    return u_file, v_file, h_file

def download_sfc_bundle(year: int, month: int, day: int, hour: int, tag: str):
    base = f"{year:04d}{month:02d}{day:02d}{hour:02d}"
    air  = fetch_netcdf_from_psl(build_psl_url_sfc("Air+Temperature", year, month, day, hour), "comphour",  f"air_{base}_{tag}.nc")
    rh   = fetch_netcdf_from_psl(build_psl_url_sfc("Relative+Humidity+%28to+100mb+only%29", year, month, day, hour), "comphour", f"rhum_{base}_{tag}.nc")
    mslp = fetch_netcdf_from_psl(build_psl_url_sfc("Pressure+at+Mean+Sea+Level", year, month, day, hour), "comphour", f"mslp_{base}_{tag}.nc")
    u10  = fetch_netcdf_from_psl(build_psl_url_sfc("10m+Zonal+Wind", year, month, day, hour), "comphour", f"u10_{base}_{tag}.nc")
    v10  = fetch_netcdf_from_psl(build_psl_url_sfc("10m+Meridional+Wind", year, month, day, hour), "vcomphour", f"v10_{base}_{tag}.nc")
    return air, rh, mslp, u10, v10

# ==================== Meteorology helpers ====================
def calculate_dewpoint_c(T_c, RH_pct):
    a, b = 17.27, 237.7
    alpha = ((a * T_c) / (b + T_c)) + np.log(RH_pct / 100.0)
    return (b * alpha) / (a - alpha)

# ==================== Product renderers ====================
def render_dewpoint_mslp_10m(ax, lon_plot, lat_sel, center_deg, mslp_sub, T_k_sub, RH_sub, u10_sub, v10_sub, params):
    data_crs = ccrs.PlateCarree(central_longitude=center_deg)

    Td_c = calculate_dewpoint_c(T_k_sub[0] - 273.15, RH_sub[0])
    Td_f = Td_c * 9/5 + 32

    cmap, norm, levels = dewpoint_colormap_and_levels()
    cf = ax.contourf(lon_plot, lat_sel, Td_f, levels=levels, cmap=cmap, norm=norm,
                     transform=data_crs, extend="both", corner_mask=True)

    # Isobars: adaptive 2–4 hPa equivalent using range
    mslp0 = np.ascontiguousarray((mslp_sub[0] / 100.0))  # Pa->hPa if needed; most 20CR prmsl already Pa
    cint = params['cint']
    mmin = np.floor(np.nanmin(mslp0) / cint) * cint
    mmax = np.ceil(np.nanmax(mslp0) / cint) * cint
    levels_m = np.arange(mmin, mmax + cint, cint)
    cs = ax.contour(lon_plot, lat_sel, mslp0, levels=levels_m, colors="black",
                    linewidths=params['mslp_lw'], transform=data_crs)
    ax.clabel(cs, fmt="%d", fontsize=8, inline=True)

    # Barbs
    LON2, LAT2 = np.meshgrid(lon_plot, lat_sel)
    si = params['stride_y']; sj = params['stride_x']
    ax.barbs(LON2[::si, ::sj], LAT2[::si, ::sj],
             u10_sub[0, ::si, ::sj], v10_sub[0, ::si, ::sj],
             length=params['barb_len'], transform=data_crs)

    cb = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, aspect=30, shrink=0.78)
    cb.set_label("Dewpoint (°F)")

def render_pl_winds(ax, lon_plot, lat_sel, center_deg, u_sub_ms, v_sub_ms, hgt_m, level_mb, params):
    data_crs = ccrs.PlateCarree(central_longitude=center_deg)

    u = u_sub_ms[0]; v = v_sub_ms[0]
    wspd_kts = np.sqrt(u**2 + v**2) * 1.94384

    if level_mb == 500:
        cmap_ws = cmap_500_wind()
        ws_levels = np.arange(20, 141, 1)
        # Convert meters → dam levels typical for 500 mb (approx range 480–600 dam)
        z = (hgt_m[0] / 10.0)
        h_levels = np.arange(480, 600, 6)
        title_level = "500 mb"
    else:
        cmap_ws = cmap_850_wind()
        ws_levels = np.arange(20, 81, 1)
        z = (hgt_m[0] / 10.0)
        h_levels = np.arange(120, 180, 3)
        title_level = "850 mb"

    cf = ax.contourf(lon_plot, lat_sel, wspd_kts, levels=ws_levels, cmap=cmap_ws,
                     extend='both', transform=data_crs)
    cs = ax.contour(lon_plot, lat_sel, z, levels=h_levels, colors='black',
                    linewidths=1.0, transform=data_crs)
    ax.clabel(cs, inline=True, fontsize=8, colors='black', fmt='%d')

    # Barbs in knots (consistent with your script)
    LON2, LAT2 = np.meshgrid(lon_plot, lat_sel)
    si = params['stride_y']; sj = params['stride_x']
    ax.barbs(LON2[::si, ::sj], LAT2[::si, ::sj],
             u[::si, ::sj] * 1.94384, v[::si, ::sj] * 1.94384,
             length=params['barb_len'], transform=data_crs)

    cb = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=30, shrink=0.78)
    cb.set_label(f'{title_level} Wind Speed (kt)')

# ==================== Main generator ====================
def generate_visualization(year, month, day, hour, region_coords, product):
    # Tag for file naming; helps avoid collisions
    tag = "st"

    # Build region & projection
    lon_w, lon_e, lat_s, lat_n = region_coords
    proj, extent_crs, extent_centered, center_deg = build_projection_and_extent(lon_w, lon_e, lat_s, lat_n)

    # Figure/axes and map cosmetics
    fig, ax = plt.subplots(figsize=(16, 10), subplot_kw={'projection': proj})
    ax.set_extent(extent_centered, crs=extent_crs)
    plot_title_time = date_title_str(int(year), int(month), int(day), int(hour))

    # ---------------- Product Branches ----------------
    if product == "Dewpoint, MSLP, and 10 m wind barbs":
        air_f, rh_f, mslp_f, u10_f, v10_f = download_sfc_bundle(int(year), int(month), int(day), int(hour), tag)

        # Read & standardize axes
        with nc.Dataset(mslp_f) as ds:
            mslp = ds.variables['prmsl'][:]
            lon0 = ds.variables['lon'][:]; lat = ds.variables['lat'][:]
        with nc.Dataset(air_f) as ds:
            T_k = ds.variables['air'][:]
        with nc.Dataset(rh_f) as ds:
            RH = ds.variables['rhum'][:]
        with nc.Dataset(u10_f) as ds:
            u10 = ds.variables['uwnd'][:]
        with nc.Dataset(v10_f) as ds:
            v10 = ds.variables['vwnd'][:]

        # Ensure lon strictly increasing 0..360
        order = np.argsort(lon0)
        lon_360 = lon0[order]
        mslp = mslp[:, :, order]; T_k = T_k[:, :, order]; RH = RH[:, :, order]
        u10 = u10[:, :, order]; v10 = v10[:, :, order]

        # Subset to region and center longitudes
        lon_sel, lat_sel, mslp_sub = subset_lon_lat(lon_360, lat, mslp, lon_w, lon_e, lat_s, lat_n)
        _,       _,       T_k_sub  = subset_lon_lat(lon_360, lat, T_k,  lon_w, lon_e, lat_s, lat_n)
        _,       _,       RH_sub   = subset_lon_lat(lon_360, lat, RH,   lon_w, lon_e, lat_s, lat_n)
        _,       _,       u10_sub  = subset_lon_lat(lon_360, lat, u10,  lon_w, lon_e, lat_s, lat_n)
        _,       _,       v10_sub  = subset_lon_lat(lon_360, lat, v10,  lon_w, lon_e, lat_s, lat_n)

        lon_plot = to_center_frame_vec(((lon_sel + 180.0) % 360.0) - 180.0, center_deg)
        lon_plot, lat_sel, mslp_sub, T_k_sub, RH_sub, u10_sub, v10_sub = ensure_increasing_axes(
            lon_plot, lat_sel, mslp_sub, T_k_sub, RH_sub, u10_sub, v10_sub
        )

        params = auto_plot_params(extent_centered, nx=lon_plot.size, ny=lat_sel.size)
        plot_base_map(ax, params)
        render_dewpoint_mslp_10m(ax, lon_plot, lat_sel, center_deg, mslp_sub, T_k_sub, RH_sub, u10_sub, v10_sub, params)
        title = f"Dewpoint, MSLP, and Wind — {plot_title_time}"

        # Cleanup
        for f in [air_f, rh_f, mslp_f, u10_f, v10_f]:
            try:
                if os.path.exists(f): os.remove(f)
            except Exception:
                pass

    elif product in ("500 mb wind & height", "850 mb wind & height"):
        level = 500 if product.startswith("500") else 850
        u_f, v_f, h_f = download_level_bundle(level, int(year), int(month), int(day), int(hour), tag)

        with nc.Dataset(u_f) as ds:
            u = ds.variables['uwnd'][0, ...]  # m/s
            lon0 = ds.variables['lon'][:]; lat = ds.variables['lat'][:]
        with nc.Dataset(v_f) as ds:
            v = ds.variables['vwnd'][0, ...]  # m/s
        with nc.Dataset(h_f) as ds:
            hgt_m = ds.variables['hgt'][0, ...]  # meters

        # Expand to [time, lat, lon] for convenience
        u3 = u[np.newaxis, ...]
        v3 = v[np.newaxis, ...]
        z3 = hgt_m[np.newaxis, ...]

        # Strictly increasing 0..360 longitudes
        order = np.argsort(lon0)
        lon_360 = lon0[order]
        u3 = u3[:, :, order]; v3 = v3[:, :, order]; z3 = z3[:, :, order]

        # Subset & center
        lon_sel, lat_sel, u_sub = subset_lon_lat(lon_360, lat, u3, lon_w, lon_e, lat_s, lat_n)
        _,       _,       v_sub = subset_lon_lat(lon_360, lat, v3, lon_w, lon_e, lat_s, lat_n)
        _,       _,       z_sub = subset_lon_lat(lon_360, lat, z3, lon_w, lon_e, lat_s, lat_n)

        lon_plot = to_center_frame_vec(((lon_sel + 180.0) % 360.0) - 180.0, center_deg)
        lon_plot, lat_sel, u_sub, v_sub, z_sub = ensure_increasing_axes(lon_plot, lat_sel, u_sub, v_sub, z_sub)

        params = auto_plot_params(extent_centered, nx=lon_plot.size, ny=lat_sel.size)
        plot_base_map(ax, params)
        render_pl_winds(ax, lon_plot, lat_sel, center_deg, u_sub, v_sub, z_sub, level, params)
        title = f"{level} mb Wind Speed & Height — {plot_title_time}"

        # Cleanup
        for f in [u_f, v_f, h_f]:
            try:
                if os.path.exists(f): os.remove(f)
            except Exception:
                pass

    else:
        raise ValueError("Unknown product selected.")

    ax.set_title(title + "\nPlotted by Sekai Chandra (@Sekai_WX)")
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    buffer.seek(0)
    plt.close(fig)
    return buffer

# ==================== Streamlit UI ====================
st.title("20th Century Reanalysis (20CR) — Global Plotter")

# Date inputs
col1, col2, col3, col4 = st.columns(4)
with col1:
    year = st.number_input("Year", min_value=1850, max_value=datetime.now().year, value=1971)
with col2:
    month = st.number_input("Month", min_value=1, max_value=12, value=2)
with col3:
    day = st.number_input("Day", min_value=1, max_value=31, value=21)
with col4:
    hour = st.number_input("Hour (UTC)", min_value=0, max_value=23, value=21)

# Region + Product
col5, col6, col7 = st.columns([1.2, 1.0, 0.7])
with col5:
    region_options = []
    for category, regions in REGIONS.items():
        for region_name in regions.keys():
            region_options.append(f"{category}: {region_name}")
    selected_region = st.selectbox("Select Region", region_options, index=0)

with col6:
    product = st.selectbox(
        "Product",
        [
            "Dewpoint, MSLP, and 10 m wind barbs",
            "500 mb wind & height",
            "850 mb wind & height",
        ],
        index=0
    )

with col7:
    generate_button = st.button("Generate", type="primary", help="Generate the 20CR visualization")

# Run
if generate_button:
    category, region_name = selected_region.split(": ", 1)
    region_coords = REGIONS[category][region_name]
    try:
        with st.spinner("Fetching 20CR (PSL) data and generating visualization..."):
            image_buffer = generate_visualization(year, month, day, hour, region_coords, product)
        st.success("Visualization generated successfully!")
        st.image(image_buffer, caption=f"{product} • {int(year)}-{int(month):02d}-{int(day):02d} {int(hour):02d} UTC • {region_name}")
        st.download_button(
            label="Download Image",
            data=image_buffer,
            file_name=f"20CR_{product.replace(' ','_').replace('/','-')}_{int(year)}{int(month):02d}{int(day):02d}{int(hour):02d}_{region_name.replace(' ', '_')}.png",
            mime="image/png"
        )
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
        st.info("Verify the date/time exists in 20CR, and that PSL is reachable. Some older hours may be unavailable.")
