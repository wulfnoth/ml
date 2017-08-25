package org.wulfnoth.ml.classification.nb

import breeze.linalg.{DenseVector, Matrix, Vector}
import com.github.fommil.netlib.BLAS
import org.wulfnoth.ml.classification.CountVector

import scala.collection.mutable

/**
  * Created by cloud on 2017/8/21.
  */
class MultinomialNBModel(map: mutable.Map[Int, CountVector],
						 dims: Int, smooth: Double)
  extends NaiveBayesModel(map, dims, smooth) {

	private val theta = {
		val ar = new Array[Double](map.size*dims)
		map.foreach(x => {
			val (label, vector) = x
			0 until dims foreach (index => {
				val attr = vector(index)
				val attrCount = if (attr == null) 0.0 else attr.sumWithWeight
				ar(label + index*map.size) = (attrCount + smooth)/(vector.sum + smooth*dims)
			})
		})
		Matrix.create(map.size, dims, ar)
	}

	private[ml] def getPro = theta

	override protected def probabilityCal(vector: Vector[Double]) = {
		val result = new Array[Double](map.size).map(_ => 1.0)
		theta.keysIterator.foreach(position => {
			//println(s"${Math.pow(proAr(position), vector(position._2))}")
			result(position._1) *= Math.pow(theta(position), vector(position._2))
		})
		//result.foreach(println)
		result.indices.foreach(index => result(index) *= postP(index))
		result
	}

}


