package org.wulfnoth.ml.utils

import breeze.linalg.DenseVector
import breeze.stats.distributions.Gaussian
import breeze.stats.{mean, stddev}
/**
  * Created by cloud on 2017/8/25.
  */
object Test {

	def main(args: Array[String]): Unit = {
//		val m1 = DenseMatrix(
//			(1.0, 2.0),
//			(3.0, 4.0)
//		)
		val v1 = DenseVector(6, 5.92, 5.58, 5.92)
		// Axis._0 纵向
		// 2.0  2.0
		val sigma = stddev(v1)
		val mu = mean(v1)
		val gaussian = new Gaussian(mu, sigma)
		printf("%.6f", gaussian.pdf(6.0))

	}

}
