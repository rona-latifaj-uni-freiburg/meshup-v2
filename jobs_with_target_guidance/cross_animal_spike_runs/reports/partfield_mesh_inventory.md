# PartField Mesh Inventory

This inventory counts the PartField-derived mesh assets currently present under:

- `jobs_with_target_guidance/partfield_features`
- `jobs_with_target_guidance/partfield_segments`

## Counting Definitions

There are two useful ways to count the work:

- **Unique source/asset meshes:** one logical mesh name, such as `blueberry`, `horse2`, or `panther`.
- **Generated PartField meshes:** colored `.ply` files created after assigning PartField buckets. The same source mesh can appear multiple times because it was curated with different bucket counts, such as 6, 7, 8, 9, 10, 12, or 20 buckets.

For presentation, the safest wording is:

> I curated PartField part decompositions for 58 domain-scoped mesh assets, producing 376 colored PartField mesh visualizations with matching label files. This includes 25 vehicle meshes, 14 SAM3D animal meshes, 6 DenseCorr3D animal meshes, chain-deformation animals, and additional dog/test animal meshes.

The domain-scoped count treats names like SAM3D `bear` and DenseCorr3D `bear` as different meshes, because they come from different datasets.

## Overall Totals

| Artifact | Count |
|---|---:|
| Feature files | 88 |
| Colored PartField `.ply` meshes | 376 |
| Matching label files | 376 |
| Summary metadata files | 131 |
| Unique name-only mesh labels | 54 |
| Unique domain-scoped mesh assets | 58 |

## Domain Breakdown

| Domain | Unique Mesh Assets | Feature Files | Colored PartField Meshes | Label Files | Summary Files |
|---|---:|---:|---:|---:|---:|
| Cars / vehicles | 25 | 48 | 176 | 176 | 46 |
| SAM3D animals | 14 | 20 | 96 | 96 | 48 |
| DenseCorr3D animals | 6 | 6 | 62 | 62 | 19 |
| Chain animals | 4 | 4 | 24 | 24 | 12 |
| Other dog/test animals | 9 | 10 | 18 | 18 | 6 |
| **Total** | **58 domain-scoped** | **88** | **376** | **376** | **131** |

## Cars / Vehicles

I curated PartField parts for **25 unique car/vehicle meshes**.

Generated artifacts:

| Artifact | Count |
|---|---:|
| Feature files | 48 |
| Colored PartField meshes | 176 |
| Label files | 176 |
| Summary files | 46 |

Bucket settings used:

| Bucket Count | Colored Meshes |
|---:|---:|
| 6 | 14 |
| 7 | 14 |
| 8 | 48 |
| 9 | 14 |
| 10 | 14 |
| 12 | 36 |
| 20 | 36 |

Unique car/vehicle assets:

`blueberry`, `bugatti_centodieci`, `cars_doc`, `f1_car`, `f1_verstappen`, `g_class`, `green_suv`, `kona`, `mini_cooper`, `no_roof_car`, `old_car`, `oldie`, `passati`, `pink_smart`, `red_car`, `red_smart`, `red_truck1`, `red_truck2`, `red_truck3`, `red_truck4`, `santa_fe`, `usa_suv`, `venom_qt`, `vintage_car`, `white_longer_mini_cargo_truck`.

Most-used car assets by number of colored PartField versions:

| Mesh | Colored PartField Versions |
|---|---:|
| `g_class` | 16 |
| `pink_smart` | 13 |
| `blueberry` | 11 |
| `f1_car` | 11 |
| `f1_verstappen` | 11 |
| `mini_cooper` | 11 |
| `red_truck1` | 11 |
| `santa_fe` | 11 |

## SAM3D Animals

I curated PartField parts for **14 unique SAM3D animal meshes**.

Generated artifacts:

| Artifact | Count |
|---|---:|
| Feature files | 20 |
| Colored PartField meshes | 96 |
| Label files | 96 |
| Summary files | 48 |

Bucket settings used:

| Bucket Count | Colored Meshes |
|---:|---:|
| 5 | 10 |
| 6 | 12 |
| 7 | 22 |
| 8 | 22 |
| 9 | 20 |
| 10 | 10 |

Unique SAM3D animal assets in colored PartField outputs:

`bear`, `bear2`, `cat`, `cat2`, `chihuahua2`, `fox`, `fox3`, `giraffe4`, `goldie`, `horse2`, `panda2`, `panda3`, `pig`, `pug`.

SAM3D animal assets by number of colored PartField versions:

| Mesh | Colored PartField Versions |
|---|---:|
| `bear` | 12 |
| `goldie` | 10 |
| `pug` | 10 |
| `fox` | 9 |
| `bear2` | 8 |
| `cat2` | 8 |
| `panda3` | 8 |
| `cat` | 5 |
| `fox3` | 5 |
| `pig` | 5 |
| `chihuahua2` | 4 |
| `giraffe4` | 4 |
| `horse2` | 4 |
| `panda2` | 4 |

The SAM3D feature cache also contains a `horse3` tensor, but the colored PartField outputs under the SAM3D folder use the mesh names listed above.

## DenseCorr3D Animals

I curated PartField parts for **6 DenseCorr3D animal meshes**.

Generated artifacts:

| Artifact | Count |
|---|---:|
| Feature files | 6 |
| Colored PartField meshes | 62 |
| Label files | 62 |
| Summary files | 19 |

Bucket settings used:

| Bucket Count | Colored Meshes |
|---:|---:|
| 5 | 6 |
| 6 | 12 |
| 7 | 12 |
| 8 | 12 |
| 9 | 10 |
| 10 | 10 |

Unique DenseCorr3D animal assets:

`bear`, `cheetah`, `elephant`, `giraffe`, `moose`, `panther`.

DenseCorr3D animal assets by number of colored PartField versions:

| Mesh | Colored PartField Versions |
|---|---:|
| `cheetah` | 14 |
| `elephant` | 11 |
| `giraffe` | 11 |
| `panther` | 11 |
| `bear` | 9 |
| `moose` | 6 |

## Chain Animals

The chain-deformation animal experiments contain **4 unique mesh assets**.

Generated artifacts:

| Artifact | Count |
|---|---:|
| Feature files | 4 |
| Colored PartField meshes | 24 |
| Label files | 24 |
| Summary files | 12 |

Unique chain animal assets:

`giraffe`, `horse2`, `horse3`, `panther`.

Bucket settings used:

| Bucket Count | Colored Meshes |
|---:|---:|
| 5 | 8 |
| 6 | 8 |
| 8 | 4 |
| 9 | 4 |

## Other Dog/Test Animals

The additional dog/test animal experiments contain **9 unique mesh assets**.

Generated artifacts:

| Artifact | Count |
|---|---:|
| Feature files | 10 |
| Colored PartField meshes | 18 |
| Label files | 18 |
| Summary files | 6 |

Unique assets:

`hippo_simple`, `hound_original`, `hound_to_bear_no_dino`, `hound_to_bulldog`, `hound_to_bulldog_no_dino`, `hound_to_cat_no_dino`, `hound_to_dachshund`, `hound_to_golden_retriever_no_dino`, `hound_to_horse_no_dino`.

Bucket settings used:

| Bucket Count | Colored Meshes |
|---:|---:|
| 8 | 2 |
| 12 | 8 |
| 20 | 6 |
| 28 | 2 |

## Best Short Slide Statement

> I curated a PartField-derived mesh-part dataset across 58 domain-scoped mesh assets. The dataset includes 25 vehicle meshes, 14 SAM3D animal meshes, 6 DenseCorr3D animal meshes, 4 chain-deformation animal meshes, and 9 additional dog/test animal meshes. Across these assets, I generated 376 colored PartField mesh visualizations, 376 matching label files, 88 cached 448D PartField feature files, and 131 metadata summaries. Each mesh was decomposed into semantic-like buckets at multiple granularities, giving reusable part labels for deformation guidance and evaluation.

