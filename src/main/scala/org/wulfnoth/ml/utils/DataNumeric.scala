package org.wulfnoth.ml.utils

import java.io.{File, FileReader}
import java.util.concurrent.atomic.AtomicInteger

import breeze.io.{CSVReader, CSVWriter}
import breeze.linalg.DenseVector
import org.wulfnoth.ml.LabeledVector
import org.wulfnoth.ml.classification.NaiveBayes

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
		val dn = new DataNumeric()
		val result = dn.transform(CSVReader.iterator(
			input = new FileReader("./data/original/original_classification.csv"),
			skipLines = 1))

		val model = new NaiveBayes().fit(dn.separate(result).toArray)

		//model.postProbability.foreach(println)

		val vector = new DenseVector[Int](Array(0,2,0,1))
//		val r = model.predictProbability(vector)
//		r.foreach(println)
		val r = model.predict(vector)
		println(r)

//		println(model.smoothProbability)
//		r.foreach(println)
//		println("---------------------")
//		model.numeric(r).foreach(println)

	}
}