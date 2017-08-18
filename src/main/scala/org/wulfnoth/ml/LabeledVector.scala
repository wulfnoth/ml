package org.wulfnoth.ml

import breeze.linalg.Vector

/**
  * Created by cloud on 2017/8/18.
  */
case class LabeledVector(label: Int, vector: Vector[Int]) {

	override def toString: String = {
		val sb = new StringBuilder
		sb.append(s"$label\t")
		vector.activeIterator.foreach(x => sb.append(s"${x._1}:${x._2}\t"))
		sb.substring(0, sb.size-1)
	}


}
