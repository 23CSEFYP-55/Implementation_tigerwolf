package org.fog.utils.distribution;

import java.util.Random;

public class PoissonDistribution extends Distribution {
    private double lambda;

    public PoissonDistribution(double lambda) {
        super();
        this.lambda = lambda;
        setRandom(new Random());
    }

    @Override
    public double getNextValue() {
        // Generate exponential inter-arrival time which corresponds to Poisson process
        double u = getRandom().nextDouble();
        return -Math.log(1.0 - u) * lambda;
    }

    @Override
    public int getDistributionType() {
        return 4; // Assuming 4 is Poisson
    }

    @Override
    public double getMeanInterTransmitTime() {
        return lambda;
    }
}
