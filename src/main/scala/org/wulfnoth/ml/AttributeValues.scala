package org.wulfnoth.ml

import scala.collection.mutable

/**
  * @author Young
  */
class AttributeValues[V] {

	private[ml] val weights = new mutable.HashMap[V, Double]()

	//private def weights(weights: mutable.HashMap[V, Double]) = this.weights = weights

	def getOrElse(value: V, default: => Double): Double =
		weights.getOrElse(value, default)

	def apply(value: V): Double = weights(value)

	def calculate(cal: Double => Double): AttributeValues[V] = {
		val result = new AttributeValues[V]
		weights.foreach(x => {result.update(x._1, cal(x._2))})
		result
	}

	def update(value: V, weight: Double): Unit =
		weights.update(value, weight)

	def incWeight(value: V): Unit = add(value, 1.0)

	def add(value: V, weight: Double): Unit = {
		val currentValue = weights.getOrElse(value, 0.0)
		weights.update(value, currentValue + weight)
	}

	def add(tuple: (V, Double)): Unit = add(tuple._1, tuple._2)

	def adds(other: AttributeValues[V]): Unit = other.weights.foreach(add)

	override def toString: String = weights.toArray.mkString("[", ":", "]")

}
