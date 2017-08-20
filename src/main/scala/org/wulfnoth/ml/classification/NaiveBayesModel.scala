package org.wulfnoth.ml.classification

import breeze.collection.mutable.SparseArray
import breeze.linalg.Vector
import org.wulfnoth.ml.AttributeValues

import scala.collection.mutable

/**
  * Created by cloud on 2017/8/18.
  */
class NaiveBayesModel(map: mutable.Map[Int, CountVector],
					 dims: Int,
					 smooth: Double) {

	private var size = map.values.map(_.length).sum

	@transient private var postConditionalP: mutable.Map[Int, SparseArray[AttributeValues[Int]]] =
		calConditionalP

	@transient private var postP: mutable.Map[Int, Double] = calP

	@transient private var smoothP: SparseArray[AttributeValues[Int]] = calSmoothP

	private def calConditionalP = {
		map.map(labelC => {
			val s = labelC._2.attributes.map(attr => {
				if (attr == null) null
				else attr.calculate(count => (count+smooth)/(labelC._2.length+smooth*attr.weights.size))
			})
			(labelC._1, s)
		})
	}

	private def calP = map.map(x => (x._1, x._2.length.toDouble/size))

	private def calSmoothP = {
		val r = CountVector(dims)
		map.foreach(x => r.add(x._2))
		r.attributes.map(attr => {
			if (attr == null) null
			else {
				val attrCount = attr.weights.values.sum
				attr.calculate(_/attrCount)
			}
			//attr.weights.map(x => (x._1, x._2/attrCount))
		})
	}

	/**
	  * 为该模型增加训练后的数据
	  * @param other 新增加的训练数据，Key为Label值，Value为该Label的统计数据
	  */
	def add(other: mutable.Map[Int, CountVector]): Unit = {
		size += other.keys.sum
		other.foreach(x => {
			if (map.contains(x._1)) map(x._1).add(x._2)
			else map += x
		})
		postConditionalP = calConditionalP
		postP = calP
		smoothP = calSmoothP
	}

	private def probabilityCal(vector: Vector[Int]) = {
		//用于存放每一个label的概率
		val result = new Array[Double](postConditionalP.size)
		postConditionalP.foreach(x => {
			var p = postP(x._1)
			(0 until vector.size).foreach(index => {
				p *= x._2(index).apply(vector(index))
			})
			result(x._1) = p
		})
		result
	}

	def predict(vector: Vector[Int]): Int = {
		val r = probabilityCal(vector)
		r.indexOf(r.max)
//		var max = -1.0
//		var index = 0
//		r.indices.foreach(v => if(r(v) > max) {index=v; max=r(v)})
//		index
	}

	def predictProbability(vector: Vector[Int]): Array[Double] = {
		val r = probabilityCal(vector)
		val sum = r.sum
		r.map(_/sum)
	}

	private[ml] def postProbability = postP

	private[ml] def postConditionalProbability = postConditionalP

	private[ml] def smoothProbability = smoothP

}
