from filters import aura, firecracker, neon, lightning, hologram, bubbles, galaxy, grid_shadow, animal, magma, prism, cyber_wings, stick_figure, extreme, boss, portal, matrix_human

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
    "Galaxy":      galaxy,
    "Grid Shadow": grid_shadow,
    "Animal":      animal,
    "Magma":       magma,
    "Prism":       prism,
    "Cyber Wings": cyber_wings,
    "Matrix Human": matrix_human,
}

FILTER_NAMES = list(FILTER_REGISTRY.keys())
