    PROGRAM PROJECT

        !The program runs all the actions 
        !for the three models: Ising Model,
        !Graph Partitioning, Vertex Cover
        !and Traveling Salesman.

    USE IM
    USE GP
    USE VC
    USE TS

    IMPLICIT NONE   

    call IsingModel()
    call GraphPartitioning()
    call VertexCover()
    call TravelingSalesman()

    STOP 
    END PROGRAM