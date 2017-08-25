package org.wulfnoth.ml.classification

import breeze.collection.mutable.SparseArray
import org.wulfnoth.ml.AttributeValues


/**
  * 用于一组属性（纬度）下各个值的计数
  */
case class CountVector(dimSize: Int) {

	//该组计数对应的Record的个数
	private var size = 0L

	//所有属性（纬度）的各个值的计数
	private[ml] val attributes = new SparseArray[AttributeValues](dimSize)

	def apply(index: Int): AttributeValues = attributes(index)

	def foreach[U](f: AttributeValues => U): Unit =
		attributes.foreach(f)

	private var sumT: Double = -1

	def sum: Double = {
		if (sumT == -1) {
			sumT = 0L
			attributes.foreach(attr => attr.weights.foreach(x => sumT += x._1*x._2))
		}
		sumT
	}

	/**
	  * 获得该Label对应的Record的个数
	  * @return 个数
	  */
	def length: Long = size

	def update(dim: Int, value: Double): Unit = {
		val attribute = attributes.getOrElseUpdate(dim, new AttributeValues)
		attribute.incWeight(value)
	}

	def inc(): Unit = size += 1

	def add(other: CountVector): Unit = {
		size += other.size
		other.attributes.iterator.foreach(x => {
			if (attributes(x._1) != null)
				attributes(x._1).adds(x._2)
			else
				attributes.update(x._1, x._2)
		})
	}

	override def toString: String = {
		val sb = new StringBuilder
		sb.append(s"size: $size\t")
		attributes.foreach(x => {
			sb.append(x.toString)
			sb.append("\t")
		})
		sb.substring(0, sb.length-1)
	}
}
