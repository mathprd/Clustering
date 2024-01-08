/*
 * Copyright (C) 2011-2016 clueminer.org
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.clueminer.clustering.benchmark.exp;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.clueminer.clustering.aggl.HCLW;
import org.clueminer.clustering.aggl.HCLWMS;
import org.clueminer.clustering.aggl.HacLwMsPar;
import org.clueminer.clustering.aggl.HacLwMsPar2;
import org.clueminer.clustering.api.AgglomerativeClustering;
import org.clueminer.clustering.benchmark.BenchParams;
import org.clueminer.clustering.benchmark.Experiment;
import static org.clueminer.clustering.benchmark.exp.Hclust.parseArguments;
import org.clueminer.log.ClmLog;

/**
 *
 * @author Tomas Barton
 */
public class HclusPar2 extends Hclust {

    /**
     * @param args the command line arguments
     */
    @Override
    public void main(String[] args) {
        BenchParams params = parseArguments(args);
        ClmLog.setup(params.log);

        benchmarkFolder = params.home + File.separatorChar + "hclust-par2";
        ensureFolder(benchmarkFolder);

        System.out.println("# n = " + params.n);
        System.out.println("=== starting experiment:");
        AgglomerativeClustering[] algorithms = new AgglomerativeClustering[]{
            new HCLW(), new HCLWMS(), new HacLwMsPar(2), new HacLwMsPar(4), new HacLwMsPar2(2), new HacLwMsPar2(4)
        };
        Experiment exp = new Experiment(params, benchmarkFolder, algorithms);
        ExecutorService execService = Executors.newFixedThreadPool(1);
        execService.submit(exp);
        execService.shutdown();
    }

}
