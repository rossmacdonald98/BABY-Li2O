import cadquery as cq
import math

def create_bucket_layer(
    cylinder_diameter: float,
    layer_thickness: float,
    bucket_diameter: float,
    wall_thickness: float,
    packing_density: float,
    hole_diameter: float = 0.0
):
    """
    Generates a single layer of cylindrical buckets in a concentric
    layout to fill a larger, user-defined cylinder.

    Args:
        cylinder_diameter (float): The diameter of the final cylindrical volume.
        layer_thickness (float): The thickness of the final layer and buckets.
        bucket_diameter (float): The outer diameter of a single bucket.
        wall_thickness (float): The thickness of the bucket walls.
        packing_density (float): The proportion of the overall area that the
                                 *internal hollow area* of the buckets makes
                                 up (0 to 1). This controls spacing.
        hole_diameter (float): The diameter of a hole in the bottom of each
                               bucket. Defaults to 0 (no hole).
    """

    # --- Input Validation ---
    if not (0 < packing_density <= 1):
        raise ValueError("Packing density must be between 0 and 1.")
    if not all(v > 0 for v in [
        cylinder_diameter, layer_thickness, bucket_diameter, wall_thickness
    ]):
        raise ValueError("All dimensions must be positive.")
    if wall_thickness >= bucket_diameter / 2:
        raise ValueError("Wall thickness must be less than half the bucket diameter.")
    if not hole_diameter >= 0:
        raise ValueError("Hole diameter cannot be negative.")

    # --- Calculations ---
    # Calculate internal bucket dimensions for packing density calculation
    inner_radius = (bucket_diameter / 2) - wall_thickness
    inner_diameter = inner_radius * 2
    if inner_radius <= 0:
        raise ValueError("Wall thickness is too large, resulting in no internal area.")
    if hole_diameter >= inner_diameter:
        raise ValueError(f"Hole diameter ({hole_diameter}mm) must be smaller than the bucket's inner diameter ({inner_diameter:.2f}mm).")


    # Density is now based on 2D area, not 3D volume
    single_bucket_internal_area = math.pi * (inner_radius**2)
    unit_cell_area = single_bucket_internal_area / packing_density
    spacing = math.sqrt(unit_cell_area)

    # --- Overlap Checks ---
    if bucket_diameter > spacing:
        print(f"Warning: Calculated spacing ({spacing:.2f}mm) is less than bucket diameter ({bucket_diameter}mm). "
              "Bucket walls will intersect as intended.")

    if spacing < inner_diameter:
        raise ValueError(
            f"Packing density is too high, causing internal volumes to overlap. "
            f"The required center-to-center spacing must be at least the inner diameter ({inner_diameter:.2f}mm), "
            f"but the current density results in a spacing of only {spacing:.2f}mm. "
            f"Please reduce the packing density or adjust dimensions."
        )

    # --- Model Generation (Optimized Additive Method) ---

    # 1. Create a single bucket to be copied
    bucket_outer_radius = bucket_diameter / 2
    outer_cylinder = cq.Workplane("XY").cylinder(layer_thickness, bucket_outer_radius)
    inner_cylinder = (
        cq.Workplane("XY")
        .workplane(offset=wall_thickness)
        .cylinder(layer_thickness, inner_radius)
    )
    bucket = outer_cylinder.cut(inner_cylinder)

    # Add a hole in the floor if specified
    if hole_diameter > 0:
        # Create a cylinder to cut the hole through the bucket floor
        hole_cutter = cq.Workplane("XY").cylinder(wall_thickness + 0.01, hole_diameter / 2) # Add tolerance
        bucket = bucket.cut(hole_cutter)


    # 2. Create an assembly and add buckets in a concentric pattern
    assembly = cq.Assembly()

    # Add the center bucket
    assembly.add(bucket, loc=cq.Location(cq.Vector(0, 0, 0)))

    # Add concentric rings of buckets
    # This radius check is the optimization: we only create buckets that
    # could possibly be inside the final cylinder.
    max_ring_radius = (cylinder_diameter / 2) + bucket_outer_radius
    ring_index = 1
    while True:
        ring_radius = ring_index * spacing
        if ring_radius > max_ring_radius:
            break

        # Calculate how many buckets fit on this ring
        circumference = 2 * math.pi * ring_radius
        num_buckets_on_ring = math.floor(circumference / spacing)

        if num_buckets_on_ring < 1:
            ring_index += 1
            continue

        # Place buckets around the ring
        angle_step = 360.0 / num_buckets_on_ring
        for i in range(num_buckets_on_ring):
            angle_deg = i * angle_step
            loc = cq.Location(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), angle_deg) * cq.Location(cq.Vector(ring_radius, 0, 0))
            assembly.add(bucket, loc=loc)

        ring_index += 1

    # 3. Union all the buckets into a single solid compound
    # This is still a slow step, but faster now with fewer objects
    unioned_buckets = assembly.toCompound()

    # 4. Create the final bounding cylinder
    bounding_cylinder = cq.Workplane("XY").cylinder(
        height=layer_thickness,
        radius=cylinder_diameter / 2
    ).val()

    # 5. Intersect the unioned buckets with the cylinder to trim the edges
    # This is the second slow step.
    final_model = unioned_buckets.intersect(bounding_cylinder)

    return final_model

# --- Parameters ---
# Define the final cylinder's dimensions
CYLINDER_DIAMETER = 140   # in mm
LAYER_THICKNESS = 5      # in mm

# Define the properties of the individual buckets
BUCKET_DIAMETER = 3      # in mm
WALL_THICKNESS = 0.4        # in mm
HOLE_DIAMETER = 0.2         # in mm. Set to 0 for no hole.
PACKING_DENSITY = 0.64       

# --- Generate the Model ---
try:
    model = create_bucket_layer(
        CYLINDER_DIAMETER,
        LAYER_THICKNESS,
        BUCKET_DIAMETER,
        WALL_THICKNESS,
        PACKING_DENSITY,
        HOLE_DIAMETER
    )

    # --- Export the Model ---
    cq.exporters.export(model, 'bucket_layer.step')
    cq.exporters.export(model, 'bucket_layer.stl')

    print("Successfully generated and exported 'bucket_layer.step' and 'bucket_layer.stl'")

except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# To view the model in CQ-editor, you can display it like this:
# show_object(model)

