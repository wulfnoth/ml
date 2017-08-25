package org.wulfnoth.ml.classification.nb

import org.wulfnoth.ml.LabeledVector
import org.wulfnoth.ml.classification.CountVector

import scala.collection.mutable

/**
  * Created by cloud on 2017/8/18.
  */
class NaiveBayes {

	private lazy val map = new mutable.HashMap[Int, CountVector]

	private var dim = 0

	private var smooth: Double = 1

	private def statisticLabel(data: Array[LabeledVector]) = {
		data.foreach(x => {
			val count = map.getOrElseUpdate(x.label, CountVector(dim))
			count.inc()
			x.vector.activeIterator.foreach(y => count.update(y._1, y._2))
		})
	}

	def smooth(s: Double): NaiveBayes = {
		if (s >= 0.0 && s <= 1.0) smooth = s
		this
	}

	def fit(data: Array[LabeledVector]): NaiveBayesModel = {
		dim = data.head.vector.size
		statisticLabel(data)
		//new MultinomialNBModel(map, dim, smooth)
		new NaiveBayesModel(map, dim, smooth)
	}

	override def toString: String = {
		map.toArray.mkString("[", ",", "]")
	}

}


