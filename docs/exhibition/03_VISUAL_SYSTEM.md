# Phase 3 — Visual Design System: Scientific Cinematic × Digital Twin
**Document ID:** `docs/exhibition/03_VISUAL_SYSTEM.md`  
**Date:** September 2026  
**Project:** AIcoScientist — The Intelligent Battery Manufacturing Lab  

---

## 1. Design Philosophy & Art Direction

AIcoScientist is a premier **Scientific Cinematic × Digital Twin** experience. It bridges the precision of advanced laboratory manufacturing equipment with the intuitive visual clarity of high-end cinematic visualization.

### 1.1 Core Tenets
- **Light-Themed Purity**: Battery research cleanrooms and advanced pilot plants are luminous, spotless, and bright. The visual atmosphere uses pearl whites, soft steels, and clean ambient light.
- **Physical Authenticity**: 3D equipment is modeled with true industrial proportions, beveled edges, mechanical tolerances, and PBR (Physically Based Rendering) materials. No low-effort glowing neon cubes or fantasy sci-fi tropes.
- **Micro-to-Macro Spatial Continuity**: Navigation is a fluid spatial journey. The camera dollies along the continuous moving electrode strip, dives directly through the calender rolls, plunges into the porous microstructure, and zooms out into the multi-dimensional AI optimization manifold.
- **Progressive Information Disclosure**:
  - *Tier 1 (Instant Intuition)*: What is this process step? (e.g., "Calendering — Compacting cathode to optimize energy density & ionic channels").
  - *Tier 2 (AI Interaction)*: What did the model recommend? (e.g., "AI selects 85 °C / 458 µm gap to maximize 5C rate retention").
  - *Tier 3 (Scientific Provenance)*: How was this verified? (Full historical sample table, confidence intervals, regret curves, DOI citations).

---

## 2. Color Palette & Thematic Tokens

The palette is engineered for high contrast, calm scientific authority, and vivid focal accents:

| Token Name | Hex Code | Role & Application |
|---|---|---|
| **Pearl White** | `#F4F7F7` | Primary background, cleanroom floor, ambient atmosphere |
| **Pristine Card**| `#FFFFFF` | Floating HUD card backgrounds (with subtle soft drop shadow) |
| **Soft Steel** | `#DCE8EC` | Structural equipment casing, frame rails, secondary borders |
| **Mint Tint** | `#E8F4F2` | Subtle badge backgrounds, safe status indicators, gentle glows |
| **Graphite Deep**| `#142A35` | Primary typography, machine chassis, dark contrast accents |
| **Graphite Slate**| `#243E4C`| Secondary text, table borders, neutral metric cards |
| **Tech Teal** | `#087F8C` | Active AI decisions, acquisition highlights, model confidence bands |
| **Teal Glow** | `#087F8C22`| Subtle glow behind active 3D targets and interactive stages |
| **Energy Orange**| `#F59E42` | Measured experimental outcomes, rediscovery celebrations, best-so-far trajectory |
| **Alert Coral** | `#E65100` | Hard constraint boundaries, warning notes, limitations badges |

---

## 3. Typography Hierarchy

- **Primary Display Font**: High-legibility modern geometric sans-serif (Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto).
  - *Hero Headlines*: 36px–48px, bold (font-weight: 700), tracking -0.02em.
  - *Stage Titles*: 20px–24px, semibold (font-weight: 600).
  - *Body Copy*: 14px–16px, regular (font-weight: 400), line-height: 1.6. Always clear and readable on 1080p and 4K displays.
- **Scientific Monospace Font**: JetBrains Mono, SF Mono, Menlo, Consolas.
  - *Applications*: Metric values (`3.029 g/cm³`, `0.7947`), Recipe IDs (`EXP_03`, `protocol-c1c280b7`), SHA-256 hashes, DOI links.
  - *Styles*: 13px–14px, medium (font-weight: 500).

---

## 4. UI Layout & Viewport Composition

```
+-----------------------------------------------------------------------------------+
|  [AIcoScientist Logo]  |  Scenario: [Drakopoulos Graphite v] [Warwick NMC622 v]   |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|                                                                                   |
|                           PERSISTENT 3D WORLD CANVAS                              |
|             (Full-Screen Interactive Three.js Scene: Equipment / Strip)           |
|                                                                                   |
|                                                                                   |
|   +-----------------------+                         +-------------------------+   |
|   | CONTEXTUAL HUD CARD   |                         | REAL-TIME METRICS CARD  |   |
|   | Current: Calendering  |                         | Density: 3.029 g/cm³    |   |
|   | Roll Gap: 458 µm      |                         | Porosity: 31.94%        |   |
|   | [Inspect Microstructure]                        | 5C Rate: 0.795 (Best)   |   |
|   +-----------------------+                         +-------------------------+   |
|                                                                                   |
+-----------------------------------------------------------------------------------+
|  << Prev Step  |   O Formulation - O Mixing - O Coating - O Calendering   |  Next >>|
+-----------------------------------------------------------------------------------+
```

- **Viewport Dominance**: 100vw × 100vh WebGL canvas as the persistent ground truth.
- **Non-Obtrusive Floating Panels**: Glass-morphism HUD cards using `backdrop-filter: blur(12px)` with 85% white opacity, rounded-2xl radii, and crisp 1px borders (`#DCE8EC`).
- **Interactive Controls**: Touch-friendly, accessible button targets (minimum 44×44px hit areas) with clear hover, active, and focus states.

---

## 5. 3D Materials & Lighting Specification

### 5.1 Industrial Materials (PBR)
1. **Calender Steel Rolls**:
   - Heavy hardened polished stainless steel (`Metalness: 0.95`, `Roughness: 0.12`).
   - Mirror-like specular highlights reflecting the cleanroom overhead light strips.
2. **Coating Doctor Blade & Slurry Die**:
   - Satin brushed aluminum (`Metalness: 0.85`, `Roughness: 0.28`).
3. **Electrode Substrate Foil**:
   - Anode: Copper foil (`Color: #D97443`, `Metalness: 0.90`, `Roughness: 0.20`).
   - Cathode: Aluminum foil (`Color: #E2E6E8`, `Metalness: 0.92`, `Roughness: 0.15`).
4. **Coated Electrode Film**:
   - Graphite film: Deep matte anthracite (`Color: #1C2321`, `Roughness: 0.88`).
   - NMC622 film: Slate gunmetal (`Color: #2C3539`, `Roughness: 0.82`).

### 5.2 Electrode Microstructure Material & Deformation
1. **Active Material Particles**:
   - Secondary spherical agglomerates and faceted grains.
   - Graphite: Oblate spheroids / lamellar flakes (`Color: #222831`).
   - NMC622: Spherical polycrystalline granules (`Color: #393E46`, subtle normal bump map).
2. **Conductive Additive & Polymeric Binder (CBD)**:
   - Web-like interconnecting matrix bridging particles (`Color: #087F8C`, semitransparent `Opacity: 0.75`).
3. **Pore Channels / Void Space**:
   - Transparent negative space. During compression, void channels narrow, particles slide and pack, and overall electrode thickness reduces by 25–40%.

### 5.3 Studio Lighting Architecture
- **Directional Sun/Key Light**: High-angle soft white light (`Color: #FFFFFF`, `Intensity: 1.4`) casting soft shadow maps across the machinery.
- **Cleanroom Ceiling Lightstrips**: Dual overhead rect-area or linear light sources (`Color: #F0F8FF`, `Intensity: 0.8`) providing long, clean specular highlights along the cylindrical rolls and equipment housings.
- **Subtle Ground Bounce**: Hemisphere light (`Sky: #FFFFFF`, `Ground: #DCE8EC`, `Intensity: 0.6`) ensuring soft shadows without harsh pitch-black recesses.
