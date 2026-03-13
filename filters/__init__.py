from filters import aura, firecracker, neon, lightning, hologram, bubbles, grid_shadow, animal, magma, prism, portal, matrix_human, infrared, kinetic_brush, flora_infusion, energy_master, gravity_pull, positive_energy, magic_spells

# Registry: display name → module with apply(canvas, pose) function
FILTER_REGISTRY = {
    "Default":      None,          # plain skeleton, no extra filter
    "Magic Spells": magic_spells,
    "Positive Aura":positive_energy,
    "Gravity Pull": gravity_pull,
    "Portal":       portal,
    "Energy Master":energy_master,
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
