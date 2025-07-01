
library(plotly)
library(dplyr)

p_Ll <- 0.25

sample_simplex <- function(n) {
  X <- matrix(rexp(3 * n), ncol=3)
  X <- P_Ll * X / rowSums(X)
  colnames(X) <- c("x", "y", "z")
  as.data.frame(X)
}

set.seed(42)
n_samples <- 50000
df <- sample_simplex(n_samples)

epsilon <- 1e-6

df <- df %>%
  mutate(
    monotonic = (x + epsilon < y) & (y + epsilon < z) & (z < P_Ll - epsilon),
    specialization = (epsilon + y+z>p_Ll),
    z_leq_xy = z <= x + y + epsilon,
    both_conditions = monotonic & z_leq_xy
  )

vertices <- data.frame(
  x = c(P_Ll, 0, 0),
  y = c(0, P_Ll, 0),
  z = c(0, 0, P_Ll),
  label = c("(P_Ll,0,0)", "(0,P_Ll,0)", "(0,0,P_Ll)")
)

plot_ly() %>%
  # All simplex points in grey
  add_markers(
    data = df,
    x = ~x, y = ~y, z = ~z,
    marker = list(color = 'darkgrey', size = 1.5),
    name = 'All simplex',
    opacity = 0.2
  ) %>%
  
  # Monotonic region in blue
  # add_markers(
  #   data = df %>% filter(monotonic),
  #   x = ~x, y = ~y, z = ~z,
  #   marker = list(color = 'blue', size = 2),
  #   name = 'Monotonic region (x < y < z)',
  #   opacity = P_Ll
  # ) %>%
  
  # Intersection in orange
  # add_markers(
  #   data = df %>% filter(both_conditions),
  #   x = ~x, y = ~y, z = ~z,
  #   marker = list(color = 'orange', size = 2),
  #   name = 'Monotonic & z ≤ x + y',
  #   opacity = 0.8
  # ) %>%
  
  add_markers(
    data = df %>% filter(specialization),
    x = ~x, y = ~y, z = ~z,
    marker = list(color = 'green', size = 2),
    name = 'specialization (p_Ll < y + z)',
    opacity =  0.8
  ) %>%
  
  # Vertices in black
  add_markers(
    data = vertices,
    x = ~x, y = ~y, z = ~z,
    marker = list(color = 'black', size = 5),
    name = 'Vertices'
  ) %>%
  
  # Vertex labels
  add_text(
    data = vertices,
    x = ~x, y = ~y, z = ~z,
    text = ~label,
    textposition = 'top center',
    showlegend = FALSE
  ) %>%
  
  layout(
    title = "3D Simplex: All (grey), Monotonic (blue), Intersection (orange)",
    scene = list(
      xaxis = list(title = "x"),
      yaxis = list(title = "y"),
      zaxis = list(title = "z"),
      camera = list(
        eye = list(x = 1.2, y = 1.2, z = 1.2)
      )
    )
  )

