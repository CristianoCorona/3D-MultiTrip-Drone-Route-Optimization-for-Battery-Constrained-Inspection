# 3D-MultiTrip-Drone-Route-Optimization-for-Battery-Constrained-Inspection
This project was developed in the context of the Fundamentals of Operations Research course at Politecnico di Milano, taught by Professors [Federico Malucelli](https://www.deib.polimi.it/ita/personale/details/67888) and [Pietro Belotti](https://www.deib.polimi.it/eng/people/details/85850). It consists of a linear programming model implemented in Python, particularly leveraging the MIP library. The specification can be found in the [dedicated file](https://github.com/CristianoCorona/3D-MultiTrip-Drone-Route-Optimization-for-Battery-Constrained-Inspection/blob/main/specs.md).

## Authors
- [Giovanni Carpenedo](https://github.com/gcarpenedo)
- [Cristiano Corona](https://github.com/CristianoCorona)
- [Giulio Dalla Costa](https://github.com/Giulio-DC03)

## General considerations on the project
We have reflected at length on the selection of the optimal base point in an attempt to separate this problem from the building structure; overall, it seems reasonable to assume that there exists an optimal way to explore the points to_visit regardless of the position of the base points, since the attack points are the only access points and can be imagined as a single large "virtual" base.

However, we have not found a satisfactory model for this approach, mainly due to capacity constraints; without knowing the base point, it is not possible to precisely constrain the drone to return to the attack points (virtual base) with an exact capacity, since it depends on the base point.

For this reason, we decided to adopt a **heuristic approach**, choosing the most promising bases and including only those in the model, thereby significantly reducing the number of base point variables (bp) and estimating an upper bound for the number of trips. We have observed that on each single arc, lower energy consumption is associated with lower time expenditure; therefore, we decided to follow this path to identify a feasible initial solution to provide to the solver in order to improve the effectiveness of Branch & Bound.

It is possible to modify the number of bases to consider in the model in the main file (at the bottom), along with the solver, the maximum optimization time, and a switch to activate/deactivate the heuristic.

We also developed another approach that does not include explicit constraints to exclude subtours in the initial model, but instead identifies and excludes them via **dynamic constraints** iteration after iteration ([dynamic_constrs.py](https://github.com/CristianoCorona/3D-MultiTrip-Drone-Route-Optimization-for-Battery-Constrained-Inspection/blob/main/dynamic_constrs.py)).

We have also included some test buildings based on the parameters of Edificio1 and some images as an example.

## How to run the script
You just have to make sure that the file you choose (main.py or dynamic_constrs.py) and the Building .csv file are in the same directory; please, note that the .csv file has to be named Edificio1.csv or Edificio2.csv according to what is defined in the specs.
Now you just have to run the script with "python *chosen_file* *chosen_building*"
