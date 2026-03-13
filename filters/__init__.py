from filters import aura, firecracker, neon, lightning, hologram, bubbles, grid_shadow, animal, magma, prism, stick_figure, extreme, boss, portal, matrix_human, infrared, kinetic_brush, flora_infusion

# Registry: display name → module with apply(canvas, pose) function
FILTER_REGISTRY = {
    "Default":      None,          # plain skeleton, no extra filter
    "Stick Figure": stick_figure,
    "Boss Mode":    boss,
    "Portal":       portal,
    "Extreme FX":   extreme,
    "Aura":         aura,
    "Neon":        neon,
    "Lightning":   lightning,
    "Firecracker": firecracker,
    "Hologram":    hologram,
    "Bubbles":     bubbles,
    "Grid Shadow": grid_shadow,
    "Animal":      animal,
    "Magma":       magma,
    "Prism":       prism,
    "Kinetic Brush": kinetic_brush,
    "Flora Infusion": flora_infusion,
    "Matrix Human": matrix_human,
    "Infrared":    infrared,
}

FILTER_NAMES = list(FILTER_REGISTRY.keys())
