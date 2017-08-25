package org.wulfnoth.ml.classification.nb

import breeze.linalg
import breeze.linalg.Matrix
import org.wulfnoth.ml.classification.CountVector

import scala.collection.mutable

/**
  * Created by cloud on 2017/8/21.
  */
class BernoulliNBModel(map: mutable.Map[Int, CountVector],
					   dims: Int,
					   smooth: Double) extends NaiveBayesModel(map, dims, smooth) {

	val theta: Matrix[Double] = {
		val ar = new Array[Double](map.size * dims)
		map.foreach(x => {
			val (label, countV) = x
			0 until dims foreach(index => {
				val count = if (countV(index) == null) 0.0 else countV(index).weights.getOrElse(1, 0.0)
				ar(label + index*map.size) = (count + smooth)/(countV.length+map.size*smooth)
			})
		})
		Matrix.create(map.size, dims, ar)
	}

	override protected def probabilityCal(vector: linalg.Vector[Double]) = {
		val result = new Array[Double](map.size).map(_ => 1.0)
		theta.foreachKey(position => {
			val (x, y) = position
			result(x) *= {if (vector(y) == 0) 1 - theta(position) else theta(position)}
		})
		result.foreach(x => printf("%.6f ", x))
		result.indices foreach (index => result(index) *= postP(index))
		result
	}

}
