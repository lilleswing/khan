""" Monte-Caro sampling of geometries using query-by-committee """

from monte_carlo.parameters import MonteCarloParams
from monte_carlo.committee import ModelCommittee, quartic_potential

import sys
import argparse
import numpy as np

def probability_density(beta, e, alpha, s):
    return np.exp(-beta*(e - (alpha/beta)*s))


def normalized_distributions(alpha, beta, test_set, committee):
    """
    Return normalized distrubutions for 
    exp(-beta e) exp(alpha s) and there product respectively
    """

    bdist, sdist, cdist = ([], [], [])
    for x, y in test_set:
        e, s = committee.predict(x)
        bdist.append(np.exp(-beta*e))
        sdist.append(np.exp(alpha*s))
        cdist.append(probability_density(beta, e, alpha, s))

    bdist = np.array(bdist)
    sdist = np.array(sdist)
    cdist = np.array(cdist)

    bdist = bdist / np.sqrt(np.dot(bdist, bdist))
    sdist = sdist / np.sqrt(np.dot(sdist, sdist))
    cdist = cdist / np.sqrt(np.dot(cdist, cdist))

    return bdist, sdist, cdist

def parse_args(args):
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Monte-Carlo sampling of high variance atomic environments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--ani-lib',
        required=False,
        help="Location of the shared object for GPU featurization"
    )

    parser.add_argument(
        '--config-file',
        default=None,
        help="config file for monte-carlo steps"
    )

    parser.add_argument(
        '--save-dir',
        default=[],
        action='append',
        type=str,
        help="Add a location of saved NN.")

    return parser.parse_args(args)

def setup_mc(sys_args):
    """
    Set up a Monte-Carlo job
    """
    args = parse_args(sys_args)
    params = MonteCarloParams()
    if args.config_file:
        params.setFromConfig(args.config_file)

    committee = ModelCommittee(nmembers=10, smax=params.getValue("sigma_max"))
    return args, params, committee

def run_monte_carlo(params, committee):
    """
    Do an MC run, this is just a model for now
    """

    params.saveConfig("config.json")

    beta = params.getValue("beta")
    alpha = -np.log(params.getValue("Pmin")) / params.getValue("sigma_max")
    dx_max = params.getValue("disp_max")

    # set func to quartic for first cycle
    func = quartic_potential

    # start in left minima
    x0 = -np.sqrt(60.0)
    samples = [] 
    high_var_data = set()
    fitting_data = set()
    n_new = 0

    # document initial values
    f0, s0 = func(x0)
    p0 = probability_density(beta, f0, alpha, s0)
    samples.append(x0)

    # sample the whole function
    test_set = [
        (x, func(x)[0]) for x in np.arange(-13.0, 13.0, 0.1)
    ]
        
    nsteps = 0
    naccepted = 0
    s_thresh = -10.0
    ibatch = 1
    for istep in range(params.getValue("max_steps")):
        # propose move
        x1 = x0 + dx_max*(2.0*np.random.random() - 1.0)
        f1, s1 = func(x1)
        p1 = probability_density(beta, f1, alpha, s1)

        # metropoulos acceptance criteria
        a = min(1.0, p1/p0)
        z = np.random.random()

        # accept
        if z < a:
            x0, f0, s0, p0 = x1, f1, s1, p1
            naccepted += 1
            # only keep new, high variance data for fitting
            if s0 > s_thresh:
                high_var_data.add((x0, f0))

        samples.append(x0)
        nsteps += 1

        batch_size = params.getValue("batch_size")
        if ibatch == 1:
            batch_size = batch_size*10

        # append samples
        if len(samples) >= batch_size:
            with open("samples.dat", "a") as fout:
                for x_example in samples:
                    fout.write("%.4f\n" % x_example)
                samples = []

        # train if we have another batch of iteresting points
        if len(high_var_data) > batch_size: 

            print("Training against batch %d after %d moves" % (ibatch, nsteps))
            print("Acceptance probability %f" % (naccepted/nsteps))

            # get reference values to fit to
            reference_data = [
                (examp, quartic_potential(examp)[0]) for examp, _ in high_var_data
            ]
     
            fitting_data.update(reference_data)
            high_var_data = set()
            ibatch += 1

            Xdata, ydata = ([], [])
            for x_example, y_example in fitting_data:
                Xdata.append([x_example])
                ydata.append(y_example)

            print("training set size %d" % len(Xdata))

            committee.train(Xdata, ydata)
            func = committee.predict
            s_thresh = params.getValue("variance_thresh")

            # update function values/density at current point
            f0, s0 = func(x0)
            p0 = probability_density(beta, f0, alpha, s0)

            # write out new samples       
            with open("new_training_data_" + str(ibatch) + ".dat", "w") as fout:
                for x_example, y_example in reference_data:
                    fout.write("%f %f\n" % (x_example, y_example))

            # write full training set
            with open("training_set_" + str(ibatch) + ".dat", "w") as fout:
                for x_example, y_example in fitting_data:
                    fout.write("%.4f %.4f\n" % (x_example, y_example))

            bdist, sdist, cdist = normalized_distributions(
                alpha,
                beta,
                test_set,
                committee
            ) 

            # data about new fit 
            rms_error = 0.0
            with open("predictions_" + str(ibatch) + ".dat", "w") as fout:
                fout.write("# x, f(x), prediction, sigma, error, exp(-b pred), exp(a s), prob density")
                for ipair, pair in enumerate(test_set):
                    x_example, y_example = pair
                    y_pred, s_pred = committee.predict(x_example)
                    err = abs(y_pred - y_example)
                    rms_error += err**2.0
                    fout.write("%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" % 
                        (x_example, y_example, y_pred, s_pred, err,
                         100*bdist[ipair], 100*sdist[ipair], 100*cdist[ipair])
                    )
                rms_error = np.sqrt(rms_error/len(test_set))

            print("Rms error %.4f" % rms_error)

    print("MC run completed")


if __name__ == "__main__":
    args, params, committee = setup_mc(sys.argv[1:])
    run_monte_carlo(params, committee)

