package org.wulfnoth.ml.classification

import breeze.collection.mutable.SparseArray

import scala.collection.mutable

/**
  * Created by cloud on 2017/8/18.
  */
class NaiveBayesModel(map: mutable.Map[Int, CountVector]) {

	private var size = map.keys.sum

	@transient private var postProbability: mutable.Map[Int, SparseArray[mutable.HashMap[Int, Double]]] = calProbability

	private def calProbability = {
		val m = new mutable.HashMap[Int, SparseArray[mutable.HashMap[Int, Double]]]
		map.foreach(x => {
			m += x._1 -> x._2.calProbability
		})
		m
	}

	def add(other: mutable.Map[Int, CountVector]) = {
		size += other.keys.sum
		other.foreach(x => {
			if (map.contains(x._1)) map(x._1).add(x._2)
			else map += x
		})
		postProbability = calProbability
	}

	def predictProbability(vector: Vector[Int]) = {
		val result = new Array[Double](postProbability.size)
		postProbability.foreach(x => {
			var p = x._1.toDouble/size
			x._2.iterator.foreach(y => p *= y._2.apply(vector(y._1)))
			//result(x._1) =
		})
	}

}
