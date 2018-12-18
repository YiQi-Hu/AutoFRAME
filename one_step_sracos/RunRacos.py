from one_step_sracos.Components import Dimension
from one_step_sracos.Racos import RacosOptimization
from one_step_sracos.ObjectiveFunction import Sphere


def test_one_step_sracos():
    # dimension setting
    dim_size = 10
    dimension = Dimension()
    dimension.set_dimension_size(dim_size)
    for i in range(dim_size):
        dimension.set_region(i, [0.0, 1.0], 0)

    budget = 100

    optimizer = RacosOptimization(dimension)

    # problem setting
    obj_fct = Sphere

    # hyper-parameter setting
    sample_size = 5
    positive_num = 2
    random_probability = 0.95
    uncertain_bit = 2

    # optimization phase
    optimizer.run_initialization(obj_fct=obj_fct, ss=sample_size, pn=positive_num, rp=random_probability,
                                 ub=uncertain_bit)

    for i in range(budget - sample_size):
        optimizer.run_one_step(obj_fct=obj_fct)

    optimal = optimizer.get_optimal()

    print('optimal value: {}'.format(optimal.get_fitness()))


if __name__ == '__main__':
    test_one_step_sracos()
