package org.wulfnoth.ml.classification.nb

import breeze.collection.mutable.SparseArray
import breeze.linalg.Vector
import org.wulfnoth.ml.AttributeValues
import org.wulfnoth.ml.classification.CountVector

import scala.collection.mutable

/**
  * Created by cloud on 2017/8/18.
  */
class NaiveBayesModel(map: mutable.Map[Int, CountVector],
							   dims: Int,
							   smooth: Double) extends Serializable{

	//训练模型使用的Record个数
	protected var size = map.values.map(_.length).sum

	//@transient private var smoothP: SparseArray[AttributeValues[Int]] = calSmoothP

	@transient private var postConditionalP: mutable.Map[Int, SparseArray[AttributeValues]] =
		calConditionalP

	@transient protected var postP: mutable.Map[Int, Double] = calP

	private def calP = map.map(x => (x._1, (x._2.length.toDouble + smooth)/(size + smooth * map.size)))

	private def updateProbability(): Unit = {
		postConditionalP = calConditionalP
		postP = calP
		//smoothP = calSmoothP
	}

	/**
	  * 为该模型增加训练后的数据
	  * @param other 新增加的训练数据，Key为Label值，Value为该Label的统计数据
	  */
	def add(other: mutable.Map[Int, CountVector]): Unit = {
		size += other.values.map(_.length).sum
		other.foreach(x => {
			if (map.contains(x._1)) map(x._1).add(x._2)
			else map += x
		})
		updateProbability()
	}

//	private def calSmoothP = {
//		val r = CountVector(dims)
//		map.foreach(x => r.add(x._2))
//		r.attributes.map(attr => {
//			if (attr == null) null
//			else {
//				val attrCount = attr.weights.values.sum
//				attr.calculate(_/attrCount)
//			}
//		})
//	}

	private def calConditionalP = {
		map.map(labelC => {
			val s = labelC._2.attributes.map(attr => {
				if (attr == null) null
				else attr.calculate(count => (count+smooth)/(labelC._2.length+smooth*attr.weights.size))
			})
			(labelC._1, s)
		})
	}

	protected def probabilityCal(vector: Vector[Double]) = {
		//用于存放每一个label的概率
		val result = new Array[Double](postConditionalP.size)
		postConditionalP.foreach( x => {
			val (label, attrs) = x
			var p = postP(label)
			(0 until vector.size).foreach(index => {
				p *= attrs(index).apply(vector(index).toInt)
			})
			result(label) = p
		})
		result
	}

	def predict(vector: Vector[Double]): Int = {
		val r = probabilityCal(vector)
		r.indexOf(r.max)
	}

	def predictProbability(vector: Vector[Double]): Array[Double] = {
		val r = probabilityCal(vector)
		val sum = r.sum
		r.map(_/sum)
	}
}
