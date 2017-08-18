package org.wulfnoth.ml.test

import breeze.linalg.{DenseVector, SparseVector}


/**
  * Created by cloud on 2017/8/16.
  */
object TestForSource {

	def main(args: Array[String]): Unit = {
//		val index = Array(1,3,5,6)
//		val values = Array(10.0, 30.0, 50.0, 60.0)
//		val sv = new SparseVector(index, values, 10)

		val dv = new DenseVector[Long](5)
		dv.foreach(println)
	}

}
