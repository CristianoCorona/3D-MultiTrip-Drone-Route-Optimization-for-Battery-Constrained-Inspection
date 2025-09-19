A company with several buildings scattered across the country has a drone for each site that analyzes the external surfaces of the building using cameras and various sensors. The set of points where the drone must stop to perform inspections for each building is known. Each point in the set is identified by three spatial coordinates; the drone’s orientation at each point can be neglected.

The drone starts from an initial point (x0, y0, z0), which must be determined (it can be chosen from a specified set described below) and where the base with the charging station will be installed. Then, the drone explores a subset of points, constrained by the remaining battery charge, returns to the starting point for recharging, and then departs again to visit other points, so as to cover all points in one or more trips.

## Drone Speeds

- Ascent speed: 1 m/s
- Descent speed: 2 m/s
- Horizontal speed: 1.5 m/s
- For oblique movements with lateral movement a and vertical (ascending) movement b, the travel time is:

  max { a / 1.5 m/s , b / 1 m/s }

The drone is assumed to be either stationary or moving at the above speeds; acceleration and deceleration times for moving between points are neglected.

## Battery Consumption

The drone has a battery capacity depending on the building and consumes energy as follows (energy per meter):

- 50 J/m for ascent
- 5 J/m for descent
- 10 J/m for lateral movements

When moving in both vertical and horizontal directions, energy consumption is the sum of the respective components.

## Problem Requirements

Choose the starting point and drone trajectories for all trips such that:

1. Every point in the given grid is covered (the drone passes over each point at least once, including starting points).
2. The drone returns to the base point for recharging whenever needed.
3. The total flight time is minimized.

## Connectivity Between Points

Two points A and B are connected if and only if:

- The Euclidean distance between A and B is at most 4 meters; OR
- The Euclidean distance is at most 11 meters and two of the coordinates x, y, or z differ by at most 0.5 meters.

This does not apply to routes between the base point and the "attack" points of the grid, which are the only points accessible to the drone when leaving from or returning to the base. The drone can traverse those paths regardless of distance.

## Problem Instances

Two instances are provided in CSV files, *Edificio1.csv* and *Edificio2.csv*, with spatial coordinates (x, y, z) of points to visit.

- **Edificio1.csv:**
  - Base point (x, y, z) must be chosen among integer coordinates with -8 <= x <= 5, -17 <= y <= -15, and z = 0.
  - Attack points are all grid points with y <= -12.5.
  - Battery capacity: 1 Wh.

- **Edificio2.csv:**
  - Base point (x, y, z) must be chosen among integer coordinates with -10 <= x <= 10, -31 <= y <= -30, and z = 0.
  - Attack points are all grid points with y <= -20.
  - Battery capacity: 6 Wh.

## Output and Visualization

A solution should be computed within a time limit of two hours. The solution must be visualized using a Python plotting library such as *matplotlib*, with each trip shown in a different color.
