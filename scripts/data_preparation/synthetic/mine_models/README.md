# 3D Mine Models

This directory contains 3D models of landmines and UXO (unexploded ordnance) for synthetic data generation.

## Model Sources

### Open Source Models

1. **Blend Swap** (CC0/CC-BY licensed):
   - https://www.blendswap.com/
   - Search: "landmine", "mine", "military equipment"

2. **Sketchfab** (Creative Commons):
   - https://sketchfab.com/
   - Filter: Downloadable, CC-BY or CC0
   - Search: "landmine", "antipersonnel mine", "antitank mine"

3. **TurboSquid Free** (Royalty-free):
   - https://www.turbosquid.com/Search/3D-Models/free/landmine
   - Free models with commercial use allowed

4. **GrabCAD** (Community models):
   - https://grabcad.com/
   - Search: "mine", "ordnance"

### Creating Custom Models

For models not available publicly:
- Use Blender to create simple geometric representations
- Focus on realistic dimensions and shapes
- Add PBR materials for realistic rendering

## Target Mine Types

Based on GICHD mine classification (8 classes):

1. **Anti-Personnel (AP) Blast Mines**
   - Examples: PMN-2, M14, Type 72
   - Typical dimensions: 6-12cm diameter, 3-5cm height
   - Usually buried just below surface

2. **Anti-Personnel (AP) Fragmentation Mines**
   - Examples: POMZ-2M, M16
   - Dimensions: 5-15cm diameter
   - Often with tripwire mechanism

3. **Anti-Tank (AT) Blast Mines**
   - Examples: TM-62M, M15
   - Dimensions: 25-35cm diameter, 10-15cm height
   - Can be partially or fully buried

4. **Anti-Tank (AT) Mines with Anti-Handling Device**
   - Similar to AT blast but with additional components
   - More complex geometry

5. **Submunitions**
   - Small cluster munitions
   - Dimensions: 3-6cm diameter
   - Various shapes (spherical, cylindrical)

6. **Improvised Explosive Devices (IED)**
   - Highly variable geometry
   - Made from common materials
   - Include containers, wires, batteries

7. **Unexploded Ordnance (UXO) - Mortar**
   - Cylindrical with fins
   - Dimensions: 6-12cm diameter, 20-40cm length

8. **UXO - Artillery Shell**
   - Larger cylindrical shape
   - Dimensions: 10-15cm diameter, 30-60cm length

## Model Requirements

Each model should:
- Be in `.blend` format or convertible (FBX, OBJ)
- Have realistic dimensions matching actual munitions
- Include UV maps for texturing
- Have clean topology (no overlapping faces, manifold mesh)
- Be centered at origin with correct scale (1 Blender unit = 1 meter)

## Material Requirements

Models should support PBR materials:
- Base Color (diffuse map)
- Metallic/Roughness
- Normal map (for surface detail)
- Optional: Ambient Occlusion

For thermal rendering:
- Models will be assigned temperature-based emission materials
- Metal parts: Lower thermal emission
- Plastic parts: Higher thermal emission
- Consider weathering and burial effects

## Directory Structure

```
mine_models/
├── README.md                    # This file
├── sources.txt                  # List of download links
├── ap_blast/
│   ├── pmn2.blend
│   ├── m14.blend
│   └── type72.blend
├── ap_fragmentation/
│   ├── pomz2m.blend
│   └── m16.blend
├── at_blast/
│   ├── tm62m.blend
│   └── m15.blend
├── submunitions/
│   └── cluster.blend
├── ied/
│   ├── simple_ied.blend
│   └── container_ied.blend
└── uxo/
    ├── mortar_81mm.blend
    └── artillery_155mm.blend
```

## Next Steps

1. Search and download CC0/CC-BY models from sources above
2. Create simple placeholder models for missing types
3. Verify all models import correctly in Blender
4. Standardize scales and orientations
5. Test rendering with different materials
