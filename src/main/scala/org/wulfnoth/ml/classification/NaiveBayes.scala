package org.wulfnoth.ml.classification

import breeze.collection.mutable.SparseArray
import org.wulfnoth.ml.LabeledVector

import scala.collection.mutable

/**
  * Created by cloud on 2017/8/18.
  */
class NaiveBayes {

	private lazy val map = new mutable.HashMap[Int, CountVector]

	private var dim = 0

	private var smooth = 1

	private def statisticLabel(data: Array[LabeledVector]) = {
		data.foreach(x => {
			val count = map.getOrElseUpdate(x.label, CountVector(dim))
			count.inc
			x.vector.activeIterator.foreach(y => count.update(y._1, y._2))
		})
	}

	def fit(data: Array[LabeledVector]): NaiveBayesModel = {
		dim = data.head.vector.size
		statisticLabel(data)

		map.foreach(println)

		null
	}

}

