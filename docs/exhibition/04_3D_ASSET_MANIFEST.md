# Phase 4 — 3D Asset Manifest & Technical Geometry Specification
**Document ID:** `docs/exhibition/04_3D_ASSET_MANIFEST.md`  
**Date:** September 2026  
**Project:** AIcoScientist — The Intelligent Battery Manufacturing Lab  

---

## 1. 3D Production Philosophy & Quality Gate

In compliance with the project's non-negotiable requirements, the 3D environment represents a **state-of-the-art battery manufacturing plant and physical electrode microstructure**. Generic low-poly colored boxes and floating spheres are prohibited. Every machine possesses:
- Real mechanical assemblies (drive motors, roll bearings, tension rollers, hydraulic cylinders, doctor blades, radiant heating elements).
- Physically plausible industrial proportions.
- Beveled edge beveling, realistic metallic and dielectric PBR materials.
- Coherent continuous electrode sheet web guiding through all stages.
- An interactive, deformation-capable electrode microstructure with particle physics and pore compression.

---

## 2. Master 3D Asset Manifest

| Asset Name | Visual Role | Geometry Breakdown | Materials / Shaders | Animation / Morphing | Classification |
|---|---|---|---|---|---|
| **`CalenderingUnit3D`** | Signature Calendering Equipment | Dual counter-rotating hardened steel rolls (dia 450mm scale), bearing chocks, hydraulic loading cylinders, digital micrometer dial, gap control servos, foil guide idle rollers. Triangles: ~32,400. | Polished stainless steel (`metalness: 0.95`, `roughness: 0.12`), painted industrial cast iron frame (`#243E4C`, `roughness: 0.45`), brass bushing sleeves. | Continuous synchronized roll rotation; hydraulic cylinder micro-adjustment; roll gap dynamic indicator. | Industrial Equipment (Digital Twin Scale) |
| **`CoatingDryingLine3D`** | Signature Coating & Drying Line | Precision slot-die head, backing roll, suspension slurry feed pipe, heated doctor blade, multi-zone convection drying tunnel with radiant IR quartz lamps and exhaust ducts. Triangles: ~28,800. | Brushed aluminum (`roughness: 0.28`), tempered glass inspection windows, glowing amber quartz lamps (`emissive: #F59E42`), copper/aluminum web. | Slurry meniscus bead flow; continuous foil traversal; pulsing heat zone indicators. | Industrial Equipment (Digital Twin Scale) |
| **`ElectrodeMicrostructure3D`** | Signature Electrode Microstructure | Multi-component porous electrode volume: 150+ discrete spherical & faceted active material particles (NMC622 / Graphite), conductive carbon-binder domain (CBD) network, copper/aluminum current collector foil. Triangles: ~48,000. | High-roughness carbonaceous active particles (`roughness: 0.85`), dark matrix web, metallic current collector base foil. | **Continuous compression morphing**: Vertical layer thickness compression (-35%), particle rearrangement & compaction, pore channel constriction. | Scientific Representation (Parametric Microstructure) |
| **`OptimizationManifold3D`** | 3D AI Optimization Search Space | Multi-dimensional response surface manifold, 3D candidate point clouds (evaluated, proposed, unvisited), uncertainty variance hulls, Pareto frontier ridges. Triangles: ~18,500. | Semitransparent gradient iso-surfaces (`opacity: 0.70`), glowing acquisition markers (`#087F8C`), historical discovery beacons (`#F59E42`). | Step-by-step acquisition beam projection; GP uncertainty envelope contraction; point selection pulse. | Data-Driven Scientific Visualization |
| **`ManufacturingLineLayout3D`** | Full Factory Scenic Environment | Cleanroom architectural envelope, resin epoxy floor with subtle specular reflection, safety rail boundaries, overhead LED lighting gantries, connecting web pathway. Triangles: ~16,200. | Epoxy resin floor (`color: #F4F7F7`, `roughness: 0.22`, `metalness: 0.05`), matte aluminum ceiling trusses, soft shadow receiving plane. | Idle subtle camera drift; traveling electrode foil ribbon. | Cleanroom Architectural Scene |

---

## 3. Signature Electrode Microstructure: Morphing Mechanics

The microstructure is the flagship physical visualization of AIcoScientist.

### 3.1 Compression Morph Architecture
- **Initial State (Uncalendered)**:
  - Film thickness: $H_0 \approx 52.5\,\mu\text{m}$ (Low loading) / $74.5\,\mu\text{m}$ (High loading).
  - High porosity: $\epsilon_0 \approx 48.0\%$.
  - Wide void percolation channels between primary active material agglomerates.
- **Deformation Function**:
  - Each particle $i$ with initial coordinate $(x_i, y_i, z_i)$ undergoes:
    $$y_i(t) = y_i(0) \cdot (1 - \lambda \cdot t) + \delta_y(i, t)$$
    $$x_i(t) = x_i(0) + \delta_x(i, t), \quad z_i(t) = z_i(0) + \delta_z(i, t)$$
    where $t \in [0, 1]$ is the user-controlled compression progress, $\lambda \approx 0.28$ is the bulk strain factor, and $\vec{\delta}(i, t)$ represents transverse particle rolling, jamming, and local compaction without geometric self-penetration.
- **Final State (Calendered)**:
  - Compressed thickness: $H_f \approx 39.1\,\mu\text{m}$ (e.g. for `EXP_03`).
  - Reduced porosity: $\epsilon_f \approx 31.9\%$.
  - Dense particle packing, creating high electronic contact area while maintaining adequate tortuous pore pathways for lithium-ion electrolyte diffusion.

---

## 4. Performance & Runtime Budget

- **Target Frame Rate**: Consistent 60 FPS on 1920×1080 desktop displays (Chrome / Edge / Firefox).
- **Total Scene Triangle Count**: ~140,000 triangles total across the full manufacturing environment, well within standard desktop WebGL limits (< 1,500,000 triangles).
- **Draw Call Optimization**: Geometry instancing for repeating structural elements (roller bearings, particle beds, lighting fixtures).
- **Memory Footprint**: < 120 MB VRAM footprint for textures and geometry buffers.
- **Graceful Degradation**: If WebGL performance drops below 30 FPS, shadow map resolution scales down dynamically and anti-aliasing adapts to maintain smooth interactivity.
