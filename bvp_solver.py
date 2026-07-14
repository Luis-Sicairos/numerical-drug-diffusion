
import numpy as np


import matplotlib.pyplot as plt


#defining the linear finite difference.
def linear_fd(lambda_, kappa, N):
    h = 1 /(N + 1)
    x = np.linspace(0,1, N+ 2)
    mainD = (-2.0 - h**2 * lambda_ / kappa)  * np.ones(N)
    offD = np.ones(N-1) 


    Alph = np.diag(mainD) + np.diag(offD, 1 ) + np.diag(offD, -1)

    b  = np.zeros(N)
    b[0] = -1

    interior = np.linalg.solve(Alph, b)
    w =np.zeros(N + 2)
    w[0] = 1
    w[-1] = 0
    w[1:-1] = interior
    return x,w

# Defining the RK4 formula

def rk4(f, x,y, h):
    k1 = f(x,y)
    k2 = f(x +h /2, y + h * k1 / 2)
    k3 = f(x + h / 2, y + h * k2 / 2)
    k4 = f(x + h, y+ h * k3)
    return y + h * (k1 + 2*k2 + 2*k3 + k4) /6  #results for RK4

#defining the Newton shooting method
def shooting_newton(lambda_, kappa, N, s0=-1.0, tol=1e-8, max_iter=100):
    h = 1/ (N+ 1)
    x = np.linspace(0, 1, N +2)
    s = s0
    errors = []

    for iteration in range(max_iter):
        y = np.array([1, s,0,1])
        uVal= [1]

        def f(xi, y):
            u = y[0]
            up = y[1]
            v =  y[2]
            vp = y[3]

            denominator_ = kappa + u 
            dfdu = lambda_ *kappa / denominator_**2

            return np.array([
                up,
                lambda_ * u / denominator_,
                vp,
                dfdu *v
            ])

        for i in range(N + 1):
            y = rk4(f, x[i], y, h)
            uVal.append(y[0])


        phi = y[0]
        dphi =  y[2]

        deltas = -phi /dphi
        s += deltas
        errors.append(abs(deltas))


        if abs(deltas) < tol or abs(phi) < tol:
            break


    return x, np.array(uVal), s, iteration + 1,errors

#defining the nonlinear finite difference newton method

def nonlinear_fd_newton(lambda_, kappa, N, tol=1e-8, max_iter=100):
    h = 1.0 /  (N +1)
    x = np.linspace(0, 1, N+ 2) #full grid length N+2


    w = np.linspace(1, 0,  N + 2)
    errors = []

    for iteration in range(max_iter):
        F   = np.zeros(N)
        J   = np.zeros((N, N))

        for i in range(N):
            wi = w[i +1]

            F[i] = (
                w[i]
                - 2 *wi
                + w[i +2]
                - h**2 * lambda_ * wi / (kappa +wi)
            )

            dt = lambda_ * kappa / (kappa +wi)**2


            J[i, i] = -2 - h**2 * dt


            if i >0:
                J[i,i -1] = 1

            if i < N - 1:
                J[i, i +1] = 1

        delta_w = np.linalg.solve(J,-F)
        w[1:-1] += delta_w

        err =np.linalg.norm(delta_w)
        errors.append(err)


        if err < tol or np.linalg.norm(F) < tol:
            break

    return x, w, iteration + 1, errors


def influx(x,u):
    return -(u[1] -u[0]) / (x[1] -x[0])

 #getting the grid refinement table
def gridrefinementtable(lambda_, kappa, N_values):
    print("\nGrid Refinement  Table")
    print("N        u(0.25)        u(0.50)        u(0.75)         J0")
    print("-" *65)

    for N in N_values:
        x, w, iterations, errors =  nonlinear_fd_newton(lambda_, kappa, N)

        u025 = np.interp(0.25, x, w)
        u050 = np.interp(0.50, x, w)
        u075 = np.interp(0.75, x, w)
        J0 = influx(x, w)

        print(f"{N:<8d} {u025:<14.8f} {u050:<14.8f} {u075:<14.8f} {J0:<14.8f}")


def generate_all_plots():
    lambda_ = 2
    kappa = 0.3
    N = 80
    N_values = [20, 40,80,160]

    # Baseline comparison plot
    xlin, wlin =linear_fd(lambda_, kappa, N)
    xshoot, ushoot,s, shootiters,shoot_errors =shooting_newton(lambda_, kappa, N)
    x_nlfd,w_nlfd,fd_iters,fd_errors = nonlinear_fd_newton(lambda_, kappa, N)

    plt.figure()
    plt.plot(xlin, wlin, label="Linear  FD")
    plt.xlabel("x" )
    plt.ylabel("u(x)" )
    plt.title("Linear fd  solution" )
    plt.legend()
    plt.grid(True)
    plt.savefig("linear_fd_solution_plot.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(xlin, wlin, label="Linear FD")
    plt.plot(xshoot,ushoot, "--", label="Nonlinear   Shooting")
    plt.plot(x_nlfd, w_nlfd, ":", label="Nonlinear FD ")
    plt.xlabel("x")

    plt.ylabel("u(x)")

    plt.title("Baseline Comparison, lambda = 2, kappa = 0.3")
    plt.legend()

    plt.grid(True)

    plt.savefig("baseline_comparison_plot.png", dpi=300)
    plt.close()

    # Grid refinement table
    gridrefinementtable(lambda_, kappa,  N_values)

    # Nonlinear method comparison
    maxdiff = np.max(np.abs(ushoot - w_nlfd))

    print("\nNonlinear Method   Comparison" )
    print("-" *40)

    print(f"Shooting Newton iterations {shootiters}")
    print(f"Nonlinear FD Newton iterations {fd_iters}" )
    print(f"Final  shooting  slope s = u'(0): {s:.10f}" )
    print(f"Maximum difference between nonlinear methods: {maxdiff:.10e}")

    # Newton converge  plots
    plt.figure()

    plt.semilogy(range(1, len(shoot_errors) + 1),shoot_errors, marker="o")
    plt.xlabel("Newton iteration" )

    plt.ylabel("Step Size Error")
    plt.title("Shooting Newton Error")
    plt.legend(["Shooting Newton"])
    plt.grid(True)
    plt.savefig("shooting_newton_error_plot.png", dpi=300 )
    plt.close()

    plt.figure()

    plt.semilogy(range(1, len(fd_errors) +1), fd_errors, marker="o" )
    plt.xlabel("Newton Iteration")
    plt.ylabel("Step Size Error" )
    plt.title("Nonlinear FD Newton Error" )
    plt.legend(["Nonlinear FD Newton"])
    plt.grid(True)
    plt.savefig("nonlinear_fd_newton_error_plot.png",  dpi=300 )
    plt.close()

    # Parameter 
    lambdaval = [0.5, 1, 2,  4]

 #Creating the plots here
    plt.figure()
    for lam in lambdaval:
        x, w, iterations, errors = nonlinear_fd_newton(lam, kappa, N)
        plt.plot(x,w,  label=f"lambda = {lam}" )

    plt.xlabel("x")

    plt.ylabel("u(x)")
    plt.title("parameter study Varying lambda with kappa = 0.3")
    plt.legend()
    plt.grid(True)
    plt.savefig("parameter_study_lambda_plot.png",  dpi=300)
    plt.close()



if __name__ ==   "__main__":
    generate_all_plots()