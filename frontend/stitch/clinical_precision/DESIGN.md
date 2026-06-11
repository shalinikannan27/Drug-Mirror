# Design System Document: Precision Clinical Editorial

## 1. Overview & Creative North Star
**Creative North Star: "The Clinical Curator"**

This design system rejects the cluttered, "dashboard-heavy" tropes of traditional pharmaceutical software. Instead, it adopts a **High-End Editorial** approach. We treat drug data with the same reverence as a luxury scientific journal. 

The design breaks the "software template" look through **intentional asymmetry** and **breathable white space**. By prioritizing a strict typographic hierarchy and tonal depth over rigid borders, we create an environment that feels authoritative yet effortless. The goal is to reduce cognitive load for researchers while projecting a persona of scientific precision and modern reliability.

---

## 2. Colors: The Tonal Spectrum
Our palette moves away from "vibrant clinical" into a more "sophisticated medical" territory.

### The "No-Line" Rule
**Explicit Instruction:** Designers are prohibited from using 1px solid borders to section off content. Boundaries must be defined solely through background color shifts or subtle tonal transitions. 
*   *Implementation:* Place a `surface_container_lowest` card on a `surface_container_low` background to define its edges.

### Surface Hierarchy & Nesting
Treat the UI as a series of physical layers—like stacked sheets of frosted glass.
*   **Background (`#f8faf9`):** The base canvas.
*   **Surface Container Low (`#f2f4f3`):** Use for large structural sidebars or secondary content areas.
*   **Surface Container Lowest (`#ffffff`):** Reserved for primary interactive cards and data entry modules to give them the highest "lift."
*   **Surface Container High (`#e6e9e8`):** Use for "recessed" elements like search bars or inactive tabs.

### The "Glass & Gradient" Rule
To elevate the experience, use **Glassmorphism** for floating elements (e.g., tooltips, popovers). Use a semi-transparent `surface` color with a `backdrop-blur` of 12px.
*   **Signature Texture:** For primary CTAs and Hero Data Points, use a subtle linear gradient from `primary` (#00694c) to `primary_container` (#008560) at a 135-degree angle. This adds "visual soul" and prevents the medical green from feeling flat or dated.

---

## 3. Typography: Authoritative Clarity
We utilize a dual-typeface system to balance technical precision with approachable modernism.

*   **Display & Headlines (Manrope):** Chosen for its geometric stability. Use `display-lg` to `headline-sm` for high-level data summaries and page titles. The wider apertures of Manrope convey openness and trust.
*   **Body & Labels (Inter):** The workhorse for research data. Inter’s tall x-height ensures readability of complex chemical names and dosages at small sizes (`body-sm`, `label-md`).

**Hierarchy Principle:** Always favor a larger font size with a lighter weight over a small, bold font. This maintains the "Editorial" feel and prevents the UI from looking "cramped."

---

## 4. Elevation & Depth: Tonal Layering
We do not use structural lines to separate data; we use physics and light.

*   **The Layering Principle:** Depth is achieved by "stacking." A `surface_container_lowest` card sitting on a `surface_container_low` section creates a soft, natural lift without the need for a legacy shadow.
*   **Ambient Shadows:** When a "floating" effect is required (e.g., a modal or a floating action button), use an extra-diffused shadow.
    *   *Spec:* `0px 12px 32px rgba(25, 28, 28, 0.06)`. The shadow color is a tinted version of `on_surface`, mimicking natural light.
*   **The "Ghost Border" Fallback:** If a border is required for accessibility (e.g., in high-contrast modes), use the `outline_variant` token at **20% opacity**. 100% opaque borders are strictly forbidden.

---

## 5. Components: Precision Primitives

### Buttons
*   **Primary:** Linear gradient (`primary` to `primary_container`), `xl` roundedness (0.75rem). No border.
*   **Secondary:** `surface_container_high` background with `on_surface` text. Feels "integrated" into the page.
*   **Tertiary:** Ghost style. No background, `primary` text. Use for low-emphasis actions like "Cancel" or "View Details."

### Cards & Lists
*   **Forbid Dividers:** Do not use horizontal lines between list items. Use 16px of vertical white space or a subtle `surface_container_low` hover state to differentiate rows.
*   **Data Cards:** Use `xl` (0.75rem) corner radius. Content should be padded with a minimum of 24px to ensure the "Editorial" breathing room.

### Form Fields (Inputs)
*   **Style:** Recessed appearance using `surface_container_high`. 
*   **States:** On focus, transition to a `ghost_border` of `primary` and a 2px outer "glow" using `primary_fixed_dim` at 30% opacity.

### Specialized Components
*   **The "Mirror" Data Visualization:** When displaying comparative drug data, use overlapping translucent shapes (using `primary` and `secondary` at 40% opacity) to create a "Venn diagram" effect that reinforces the brand name and purpose.
*   **Status Badges:** Use `sm` (0.125rem) roundedness for a sharper, more "technical" feel compared to rounded buttons.

---

## 6. Do’s and Don’ts

### Do
*   **Do** use asymmetrical margins (e.g., a wider left margin for titles) to create an editorial flow.
*   **Do** use `on_surface_variant` (#3d4943) for secondary text to maintain high contrast while softening the overall look.
*   **Do** utilize `tertiary` (#993f3a) sparingly for critical warnings or "Contraindication" alerts.

### Don't
*   **Don't** use pure black (#000000). Use `on_background` (#191c1c) for all dark text to maintain the charcoal warmth.
*   **Don't** use standard "Drop Shadows." Only use the Ambient Shadow spec defined in Section 4.
*   **Don't** crowd the interface. If a screen feels "busy," increase the `surface` spacing rather than adding more borders or boxes.