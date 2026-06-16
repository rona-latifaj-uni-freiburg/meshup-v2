# Study Project Updates

18/05/2026:

Planned for upcoming week:

* Continue benchmarking the PartField-guided pipeline more systematically.
* Compare hard bucket Chamfer, soft PartField matching, and hybrid guidance across cars, dogs, and other mesh categories.
* Prepare clearer visualizations and videos showing how parts move during deformation.

Done:

* Extended the PartField target-guidance pipeline beyond hard bucket-only matching.
* Added soft PartField guidance, where correspondence can be guided by PartField feature similarity instead of depending only on a fixed number of hard buckets.
* Added hybrid PartField guidance, combining:
    * hard bucket Chamfer between corresponding PartField regions
    * soft PartField feature matching for more flexible semantic correspondence
* Ran new experiments on dogs and new car meshes, including Formula 1 cars, trucks, and SUVs.
* Created PartField-colored mesh visualizations and synchronized turntable videos to better inspect how final deformations compare to the target mesh.
* Researched possible benchmark setups for evaluating target-mesh deformation quality and semantic part preservation.

Results / Observations:

* Hybrid and soft PartField guidance generalized better than hard bucket-only matching.
* The dog deformation improved, especially around the previous ear failure case.
* The pipeline also generalized well to the new Formula 1 cars, trucks, and SUV examples.
* The visualizations make it easier to see whether the correct semantic regions are moving toward the corresponding target regions.

12/05/2026:

Planned for upcoming week:

* Test whether using PartField at multiple levels of detail improves target matching.
* Use coarser buckets early in optimization and finer buckets later for detail refinement.
* Create more challenging car-to-car target mesh examples to test generalization.

Done:

* Implemented multi-scale PartField Chamfer guidance.
* Used fewer buckets at the beginning of optimization and more buckets later, so the deformation first matches coarse structure and then refines details.
* Ran ablations to study the effect of bucket count and bucket scheduling.
* Created new car meshes for additional tests, including Formula 1 cars, trucks, Mini Cooper / G-Class style SUV cases, and other SUV examples.
* Fixed a mesh preprocessing issue where disconnected mesh components caused problems; the new car meshes were made connected before running the target-guidance experiments.

Results / Observations:

* Multi-scale buckets gave a more controlled coarse-to-fine deformation process.
* Starting with fewer buckets helped avoid over-constraining the early deformation.
* Finer buckets later helped preserve and match more detailed part structure.
* The connected-mesh preprocessing made the new car experiments more stable.

05/05/2026:

Planned for upcoming week:

* Replace naive spatial part zones with semantic PartField buckets.
* Run ablations to evaluate how the number of buckets affects deformation quality.
* Test whether PartField-based Chamfer generalizes better across mesh categories.

Done:

* Integrated PartField into the target mesh guidance pipeline.
* Used PartField to segment meshes into semantically meaningful buckets.
* Used the resulting PartField bucket labels to compute Chamfer loss only between corresponding parts.
* Ran ablations to test the impact of different bucket counts and PartField settings.
* Tested the method on cars and dogs.

Results / Observations:

* PartField buckets produced better semantic part matching than the earlier naive 3D spatial zones.
* The car deformations stayed more consistent because wheels, body, cabin, and other regions were pulled toward corresponding target regions.
* The method generalized better than the hand-written car-specific part zones.
* It also worked well on dogs overall, though the ears were still problematic in the first hard-bucket version.

28/04/2026:

Planned for upcoming week:

* Implement PartField for mesh part segmentation.
* Use the resulting part labels to compute Chamfer loss only between corresponding parts.
* Evaluate whether PartField part-aware Chamfer improves matching quality and generalizes to animals and other mesh types.

Done:

* Implemented a first version of part-aware Chamfer loss:
    * current version divides the mesh into naive 3D spatial zones
    * normalizes source and target meshes into the same coordinate scale before assigning zones
    * for cars, creates 12 rough buckets such as front_left_lower_wheel_zone, rear_trunk_upper, front_bumper, left_side_body, and right_side_body
    * during optimization, Chamfer is computed between matching zones only, so for example source rear_trunk_upper vertices are only pulled toward target rear_trunk_upper vertices

Results / Observations:

* Chamfer loss helps the deformation move much closer to the target mesh shape.
* Current limitation: the regions are still naive 3D zones, so this is not yet true semantic part matching. We plan to resolve this with mesh part segmentation, which could then be used for any type of mesh.
* Some target-specific geometry is still hard to generate due to limitations in the generative model, such as rear spoilers on sports cars.

21/04/2026:

Planned for upcoming week:

* Upon discussion with Artur, we decided to integrate Chamfer loss for the next week.

Done:

* Tested different scheduling strategies for balancing target mesh guidance with SDS guidance during deformation:
    * cosine scheduling
    * three-stage scheduling
* Researched ways to make the target mesh influence the shape more strongly:
    * Chamfer loss
    * silhouette / mask-based shape loss
    * ARAP regularization
    * contrastive target loss

Results / Observations:

* The three-stage schedule worked best overall.
    * In the early stage, target mesh guidance helps establish the coarse target shape.
    * In the later stage, SDS is more useful for refinement and prompt-consistent details.
