package org.wulfnoth.ml

import scala.collection.mutable

/**
  * @author Young
  */
class AttributeValues {

	private[ml] val weights = new mutable.HashMap[Double, Double]()

	//private def weights(weights: mutable.HashMap[V, Double]) = this.weights = weights

	def getOrElse(value: Int, default: => Double): Double =
		weights.getOrElse(value, default)

	def apply(value: Int): Double = weights(value)

	def calculate(cal: Double => Double): AttributeValues = {
		val result = new AttributeValues
		weights.foreach(x => {result.update(x._1, cal(x._2))})
		result
	}

	def update(value: Double, weight: Double): Unit =
		weights.update(value, weight)

	def incWeight(value: Double): Unit = add(value, 1.0)

	private[ml] def sumWithWeight = {
		var sum = 0.0
		weights.foreach(x => sum += x._1*x._2)
		sum
	}

	def add(value: Double, weight: Double): Unit = {
		val currentValue = weights.getOrElse(value, 0.0)
		weights.update(value, currentValue + weight)
	}

	def add(tuple: (Double, Double)): Unit = add(tuple._1, tuple._2)

	def adds(other: AttributeValues): Unit = other.weights.foreach(add)

	override def toString: String = weights.toArray.mkString("[", ":", "]")

}
