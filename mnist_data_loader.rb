require 'zlib'
require 'byebug'
require 'csv'

class MNIST

	def self.load_training_data
		load('./data/mnist_train.csv')
	end

	def self.load_test_data
		load('./data/mnist_test.csv')
	end

	def self.load(filename)
		images,values = [],[] 
		CSV.foreach(filename, :headers => false).each do |row|
      values.push(value_to_output(row[0]))
      images.push(row[1..-1].map(&:to_f))
		end
		[images,values]
	end

	def self.value_to_output(value)
		v = Array.new(10,0)
		v[value.to_i] = 1
		v
	end
end