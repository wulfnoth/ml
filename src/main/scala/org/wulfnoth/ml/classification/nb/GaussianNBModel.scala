package org.wulfnoth.ml.classification.nb

import breeze.linalg
import breeze.linalg.DenseVector
import breeze.stats.distributions.Gaussian
import breeze.stats.{mean, stddev}
import org.wulfnoth.ml.AttributeValues
import org.wulfnoth.ml.classification.CountVector

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * Created by cloud on 2017/8/25.
  */
class GaussianNBModel(map: mutable.Map[Int, CountVector],
					  dims: Int) extends NaiveBayesModel(map, dims, 1.0) {

	private def transform(attr: AttributeValues) = {
		val ab = new ArrayBuffer[Double]
		attr.weights.foreach(x => {
			val (count, value) = x
			0 until count.toInt foreach (ab += value)
		})
		val denseVector = DenseVector(ab.toArray)
		new Gaussian(mean(denseVector), stddev(denseVector))
	}

	val gaussiansWithLabel: mutable.Map[Int, Array[Gaussian]] = map.map(x => {
		val (label, countV) = x
		val gaussians = new Array[Gaussian](dims)
		countV.attributes.map(attr => {
			if (attr == null) new Gaussian(0, 1)
			else transform(attr)
		})
		(label, gaussians)
	})

	override protected def probabilityCal(vector: linalg.Vector[Double]) = {
		val result = new Array[Double](map.size).map(_ => 1.0)
		result.indices.foreach(index => {
			val gs = gaussiansWithLabel(index)
			gs.indices.foreach(dim => {
				result(index) *= gs(dim).pdf(vector(dim))
			})
		})
		result.foreach(println)
		result.indices.foreach(index => result(index) *= postP(index))
		result
	}


}
