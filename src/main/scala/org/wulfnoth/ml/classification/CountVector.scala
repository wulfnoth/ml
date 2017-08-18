package org.wulfnoth.ml.classification

import breeze.collection.mutable.SparseArray

import scala.collection.mutable

/**
  * Created by cloud on 2017/8/18.
  */
case class CountVector(dimSize: Int) {

	private var size = 0L

	private val values = new SparseArray[mutable.Map[Int, Long]](dimSize)

	def length: Long = size

	def update(dimensionality: Int, value: Int): Unit = {
		val map = values.getOrElseUpdate(dimensionality, new mutable.HashMap[Int, Long]())
		val count: Long = map.getOrElse(value, 0)
		map.update(value, count+1)
	}

	def inc(): Unit = size += 1

	def add(other: CountVector): Unit = {
		size += other.size
		other.values.iterator.foreach(x => {
			if (values.isActive(x._1)) {
				x._2.foreach(y => {
					val count = values(x._1).getOrElse(y._1, 0)
					values(x._1).update(y._1, count + y._2)
				})
			} else {
				values.update(x._1, x._2)
			}
		})
	}

		def calProbability = {
			values.map(x => {
				if (x != null) {
					val map = new mutable.HashMap[Int, Double]()
					x.foreach(y => map.update(y._1, y._2.toDouble / size))
					map
				} else {
					null
				}
			})
		}

	override def toString: String = {
		val sb = new StringBuilder
		sb.append(s"size: $size\t")
		values.foreach(x => {
			sb.append(x.mkString("[", "\t", "]"))
			sb.append("\t")
		})
		sb.substring(0, sb.length-1)
	}
}
