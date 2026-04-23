#import "@preview/tyniverse:0.2.3": homework

#show: homework.template.with(
  course: "Computer Vision",
  number: 3,
  student-infos: ((name: "葉騏緯", id: "ID: 114598085"),),
)
#show math.equation: set text(font: "New Computer Modern Math")

#let question = homework.complex-question

#question[
  *Grayscale + Gaussian Blur*: Convert each input image to grayscale and apply Gaussian smoothing. Save the result as `img?_q1.png`.
]

== Algorithm

Q1 reuses the grayscale conversion from HW1 and the Gaussian filter from HW2. In this homework, its job is just to prepare a stable input for Canny. The main concern is blur strength: enough to remove small texture and noise, but not so strong that the card boundary becomes hard to localize.

== Implementation

This stage directly reuses the earlier grayscale and Gaussian modules, then saves the smoothed grayscale image for Q2.

The selected parameter setting is the same for all three images:
$
  r = 2, quad k = 2r + 1 = 5, quad sigma = 1.4.
$
A $5 times 5$ kernel with $sigma = 1.4$ is a compromise. Weaker blur leaves too many short noisy edges. Stronger blur spreads the card boundary and weakens later localization. Since the three images have similar scale and resolution, one shared setting is enough.

#question[
  *Canny Edge Detection*: Use the Q1 result as input and detect edge pixels. Save the result as `img?_q2.png`.
]

== Algorithm

Canny edge detection has four parts.

1. *Input smoothing.* Since Q1 already smooths the image, no extra blur is used here. A second blur would spread the boundary further and reduce NMS accuracy.

2. *Gradient computation.* Sobel operators are used to compute the horizontal and vertical gradients:
$
  G_x =
  mat(
    -1, 0, 1;
    -2, 0, 2;
    -1, 0, 1;
  )
  * f,
  quad
  G_y =
  mat(
    -1, -2, -1;
    0, 0, 0;
    1, 2, 1;
  )
  * f.
$
The gradient magnitude is
$
  G = sqrt(G_x^2 + G_y^2).
$

3. *Non-maximum suppression.* The gradient direction is quantized into four cases: horizontal, vertical, diagonal down, and diagonal up. Each pixel is compared with the two neighbors along that direction and kept only if it is a local maximum:
$
  G(x, y) >= G_"before" quad "and" quad G(x, y) >= G_"after".
$
The non-strict $>=$ condition keeps tied pixels. This slightly favors edge continuity. At the image boundary, out-of-bounds neighbors are treated as zero in NMS.

4. *Double thresholding and hysteresis.* Pixels above the high threshold are strong edges, pixels between the two thresholds are weak edges, and the rest are removed. A weak edge is kept only if it connects to a strong edge in the 8-neighborhood.

== Implementation

All Canny steps are custom implementations. Both the Sobel gradient step and the NMS step use the same interior/border split structure. The interior loop uses direct row access without bounds checking, while only the four border strips use the padding-aware path. Since border pixels are only $O(sqrt(n))$ for an $n$-pixel image, the dominant $O(n)$ path stays branch-free.

In all three cases, the low and high thresholds keep a $1:2$ ratio. This keeps only clear strong edges at the high threshold, while still allowing weaker but real boundary pixels to survive through hysteresis.

#table(
  columns: (auto, auto, auto, 1fr),
  inset: 6pt,
  align: horizon,
  [Image], [Low], [High], [Reason],
  [`img1`],
  [$55$],
  [$110$],
  [The card boundary is clear, so a slightly higher threshold suppresses background texture while preserving the main rectangular edge.],

  [`img2`],
  [$55$],
  [$110$],
  [The input has similar edge strength to `img1`; keeping the same threshold gives clean edge maps and avoids extra weak lines.],

  [`img3`],
  [$50$],
  [$100$],
  [The boundary is weaker, so the thresholds are lowered to keep enough card-edge pixels for Hough voting.],
)

#question[
  *Hough Transform + Draw the Line*: Detect the main card boundary from the Q2 edge image and draw the final lines on the original RGB image. Save the result as `img?_q3.png`.
]

== Algorithm

A rectangular card has two pairs of parallel sides. The goal here is not just to find strong lines, but to recover the four lines that form the card boundary.

The Hough transform first moves the problem into parameter space. A line in image coordinates is represented by
$
  rho = x cos theta + y sin theta.
$
Each edge pixel votes for candidate $(rho, theta)$ pairs. Peaks in the accumulator correspond to likely lines. In this work, the accumulator uses $Delta rho = 1$ pixel and $Delta theta = 1 degree$.

Raw Hough output still contains many extra lines, so a second stage is used to select the card border:

1. *Angle clustering.* Since angle is circular, $0$ and $pi$ mean the same direction. A direct split is unreliable near the wraparound point. To avoid this, an angle histogram is built, the sparsest bin is used as the cut point, the angles are unwrapped from that point, and the sorted angles are split at the largest gap. This gives two direction groups.

2. *Outer-line selection.* In each group, only the two most extreme $rho$ values are kept. These should correspond to the two card sides. Middle lines are usually duplicates or interior responses.

3. *Corner computation.* The four corners are the pairwise intersections between the two groups. Two Hough lines
$
  rho_a = x cos theta_a + y sin theta_a,
  quad
  rho_b = x cos theta_b + y sin theta_b
$
give the linear system
$
  mat(
    cos theta_a, sin theta_a;
    cos theta_b, sin theta_b;
  )
  mat(x; y)
  =
  mat(rho_a; rho_b).
$
By Cramer's rule, letting $D = cos theta_a sin theta_b - sin theta_a cos theta_b$,
$
  x = frac(rho_a sin theta_b - rho_b sin theta_a, D),
  quad
  y = frac(rho_b cos theta_a - rho_a cos theta_b, D).
$
If $|D|$ is near zero, the two lines are nearly parallel and the intersection is rejected.

4. *Quadrilateral validation.* The four corners are sorted around the center. The shape must be convex, large enough, and close enough to the image boundary to be a reasonable card candidate. If the initial $rho$-span condition is too strict, the span threshold is progressively reduced as a fallback.

5. *Drawing.* The final result draws four vertex-to-vertex segments, not full infinite Hough lines.

If the quadrilateral validation fails, the method falls back to drawing the strongest detected Hough lines.

== Implementation

The Hough accumulator, local maximum detection, and card-border selection are all custom implementations. During voting, every edge pixel must be tested against every $theta$ bin. To avoid calling `cos()` and `sin()` inside this innermost loop, all trigonometric values are precomputed into lookup tables before voting starts. This removes the main per-iteration cost and keeps the accumulator update loop much cheaper.

Per-image Hough parameters:

#table(
  columns: (auto, auto, auto, 1fr),
  inset: 6pt,
  align: horizon,
  [Image], [Vote threshold], [Suppression radius], [Reason],
  [`img1`],
  [$105$],
  [$8$],
  [Votes are strong, so a higher threshold can remove short false lines. Radius $8$ also reduces duplicate detections from the same edge.],

  [`img2`],
  [$105$],
  [$4$],
  [The card is more tilted, so nearby valid candidates appear more often. A smaller radius keeps them for later clustering.],

  [`img3`],
  [$100$],
  [$8$],
  [Edge responses are weaker, so the threshold is lowered slightly to keep enough card-side votes.],
)

Quadrilateral validation parameters:

#table(
  columns: (auto, auto, 1fr),
  inset: 6pt,
  align: horizon,
  [Parameter], [Value], [Rationale],
  [Angle bin width],
  [$10 degree$],
  [Large enough to merge near-parallel duplicates, but still separates the two card directions.],

  [Min. group angle gap], [$15 degree$], [Prevents two almost parallel groups from being accepted as a rectangle.],
  [Min. $rho$ span], [$50$ px], [Rejects duplicate responses from the same side.],
  [Vertex tolerance], [$80$ px], [Allows corners to lie slightly outside the frame.],
  [Min. area ratio], [$2%$], [Rejects very small or degenerate quadrilaterals.],
)

#question[
  *Results*: Show the three input images and all nine output images.
]

#figure(
  caption: [Input images and HW3 outputs. Each row shows one test image, its Q1 grayscale + Gaussian result, its Q2 Canny edge result, and its Q3 Hough-line result.],
  table(
    columns: (1fr, 1fr, 1fr, 1fr),
    inset: 4pt,
    align: horizon,
    [Input], [Q1: grayscale + Gaussian], [Q2: Canny edges], [Q3: Hough lines],
    [
      #image("project_hw3/test_imgs/img1.png", width: 100%)
      #text(size: 8.5pt)[`img1.png`]
    ],
    [
      #image("project_hw3/result_imgs/img1_q1.png", width: 100%)
      #text(size: 8.5pt)[`img1_q1.png`]
    ],
    [
      #image("project_hw3/result_imgs/img1_q2.png", width: 100%)
      #text(size: 8.5pt)[`img1_q2.png`]
    ],
    [
      #image("project_hw3/result_imgs/img1_q3.png", width: 100%)
      #text(size: 8.5pt)[`img1_q3.png`]
    ],

    [
      #image("project_hw3/test_imgs/img2.png", width: 100%)
      #text(size: 8.5pt)[`img2.png`]
    ],
    [
      #image("project_hw3/result_imgs/img2_q1.png", width: 100%)
      #text(size: 8.5pt)[`img2_q1.png`]
    ],
    [
      #image("project_hw3/result_imgs/img2_q2.png", width: 100%)
      #text(size: 8.5pt)[`img2_q2.png`]
    ],
    [
      #image("project_hw3/result_imgs/img2_q3.png", width: 100%)
      #text(size: 8.5pt)[`img2_q3.png`]
    ],

    [
      #image("project_hw3/test_imgs/img3.png", width: 100%)
      #text(size: 8.5pt)[`img3.png`]
    ],
    [
      #image("project_hw3/result_imgs/img3_q1.png", width: 100%)
      #text(size: 8.5pt)[`img3_q1.png`]
    ],
    [
      #image("project_hw3/result_imgs/img3_q2.png", width: 100%)
      #text(size: 8.5pt)[`img3_q2.png`]
    ],
    [
      #image("project_hw3/result_imgs/img3_q3.png", width: 100%)
      #text(size: 8.5pt)[`img3_q3.png`]
    ],
  ),
)
