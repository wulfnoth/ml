package org.wulfnoth.ml.utils

import java.io.{File, FileReader}
import java.util.concurrent.atomic.AtomicInteger

import breeze.io.{CSVReader, CSVWriter}
import breeze.linalg.DenseVector
import breeze.stats.distributions.Gaussian
import org.wulfnoth.ml.LabeledVector
import org.wulfnoth.ml.classification.CountVector
import org.wulfnoth.ml.classification.nb.{BernoulliNBModel, MultinomialNBModel, NaiveBayes}

import scala.collection.{immutable, mutable}
import scala.collection.mutable.ArrayBuffer

/**
  * Created by cloud on 2017/8/18.
  */
class DataNumeric {

	class Dict {
		val dict = new mutable.HashMap[String, Int]()
		val id = new AtomicInteger()

		def get(key: String): Int = dict.getOrElseUpdate(key, id.getAndIncrement())
	}

	private val dics = new ArrayBuffer[Dict]

	private var initial = false

	def transform(iterator: Iterator[IndexedSeq[String]]): immutable.IndexedSeq[IndexedSeq[Int]] = {

		iterator.map(x => {
				var count = 0
				x.map(y => {
					if (count >= dics.size) dics += new Dict
					count += 1
					dics(count-1).get(y)
				})
			}).toIndexedSeq
	}

	def separate(data: IndexedSeq[IndexedSeq[Int]]): IndexedSeq[LabeledVector] = {
		data.map(x => {
			val ar: Array[Int] = x.slice(0, x.length-1).toArray
			LabeledVector(x.last, new DenseVector[Int](ar))
		})
	}

}

object DataNumeric {

	def main(args: Array[String]): Unit = {
//		val dn = new DataNumeric()
//		val result = dn.transform(CSVReader.iterator(
//			input = new FileReader("./data/original/original_classification.csv"),
//			skipLines = 1))
//
//		val model = new NaiveBayes().smooth(0.0).fit(dn.separate(result).toArray)
//
//		val vector = new DenseVector[Int](Array(0,2,0,1))
//		val r = model.predictProbability(vector)
//		r.foreach(println)
//		val r = model.predict(vector)
//		println(r)

//		val c_0 = CountVector(6)
//		c_0.inc()
//		c_0.inc()
//		c_0.inc()
//		c_0.update(0,1)
//		c_0.update(0,1)
//		c_0.update(0,1)
//		c_0.update(1,1)
//		c_0.update(2,1)
//		c_0.update(3,1)
//		val c_1 = CountVector(6)
//		c_1.inc()
//		c_1.update(0,1)
//		c_1.update(4,1)
//		c_1.update(5,1)
//		val map = new mutable.HashMap[Int, CountVector]{
//			+=(0 -> c_0)
//			+=(1 -> c_1)
//		}
		//val m = new MultinomialNBModel(map, dims = 6, smooth = 1.0)
		//val m = new BernoulliNBModel(map, dims = 6, smooth = 1.0)

//		val matrix = m.theta
//		var row = 0
//		matrix.keysIterator.foreach(position => {
//			if (position._1 > row) {
//				println()
//				row += 1
//			}
////			println(position)
//			printf("%.6f ", matrix(position))
//		})
//		println()

		val c_0 = CountVector(3)
		c_0.update(0, 6.0)
		c_0.update(0, 5.92)
		c_0.update(0, 5.58)
		c_0.update(0, 6)

//		val m = new Gaussian(map, 3)
//		val r = m.predictProbability(new DenseVector[Int](Array(3, 0, 0, 0, 1, 1)))
//		r.foreach(println)
	}
}