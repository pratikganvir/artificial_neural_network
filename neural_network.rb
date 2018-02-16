class NeuralNetwork
  include Math
  require 'byebug'
  require 'matrix'
  require './mnist_data_loader'
  attr_accessor :sizes, :input, :output, :weights, :biases, :cost_function, :net

  def initialize(sizes,input,output,cost_function=nil,weights=nil,biases=nil)
    @sizes = sizes
    @weights = weights || initialize_values
    @biases = biases || initialize_values
    @input = input
    @output = output
    @cost_function = cost_function || :sigmoid
    @activations = {}
    @derivatives = {}
    @net = {}
  end

  def initialize_values
    values = []
    @sizes.each_cons(2) do |l1,l2|
      values << Array.new(l2) {|a1| Array.new(l1) {|i| rand/100 }}
    end
    values
  end

  def train
    @input.each_with_index do |input,index_1|
      if !index_1.zero? && ((index_1%600) == 0)
        puts "#{(index_1*100)/(@input.count)}% complete"
      end
      @activations[0] = input.normalize
      # puts "*************** Weights **********************"
      # puts @weights.map { |x| x.join(',') }
      @weights.each_with_index do |weights_v,index_2|
        feedforward(input,weights_v,index_1,index_2)
      end
      backprop(input,index_1)
    end
  end

  def feedforward(input,weights_v,index_1,index_2)
    if weights_v.two_dimensional?
      z = hadmard_product(weights_v,@activations[index_2])
      net = z.map {|z1| z1.sum }
      y = z.map {|z1| send(@cost_function, z1.sum) }
    else
      z = hadmard_product(weights_v,@activations[index_2])
      net = z.map {|z1| z1.sum }
      y = z.map {|z1| send(@cost_function, z1) }
    end
    ydash = derivative(y)
    @activations[index_2+1] = y
    @derivatives[index_2] = ydash
    @net[index_2] = net
  end

  def backprop(input,index_1)
    dy_do = array_substraction(@activations[@weights.count],@output[index_1])
    # puts "*************** Error **********************"
    # puts dy_do.join(',')
    @activations.keys.reverse.each do |akey|
      break if akey.to_i.zero?
      if (@weights.count > 1) && akey!=@weights.count
        dy_dw = @weights[akey.to_i].transpose.map.with_index {|w1,i| w1.zip(dy_do).map {|w2,w3| w2*w3 }.sum}
        dy_dw = @derivatives[akey.to_i-1].zip(dy_dw).map {|w1,w2| w1*w2}
        dy_di = @activations[akey-1]
        weight_difference = dy_dw.map {|d1| dy_di.map {|d2| d2*d1}} #dy_do*dy_dw*dy_di
      else
        dy_dw = hadmard_product(dy_do,@derivatives[akey-1])
        dy_di = @activations[akey-1]
        weight_difference = dy_di.map {|a| dy_dw.map {|b| b*a}}.transpose #dy_do*dy_dw*dy_di
      end
      @weights[akey.to_i-1] = array_substraction(@weights[akey.to_i-1],weight_difference)
    end
  end

  def test
    index_1 = (0..(@input.count-1)).to_a.sample
    input = @input[index_1]
    @activations[0] = input.normalize
    @weights.each_with_index do |weights_v,index_2|
      feedforward(input,weights_v,index_1,index_2)
    end
    received_output = @activations[@activations.keys.sort.last].map {|o| (o > 0.5) ? 1 : 0}
    expected_output = @output[index_1]
    puts "Received output: #{received_output}"
    puts "Expected output: #{expected_output}"
    puts received_output == expected_output ? "Pass" : "Fail"
  end

  ##################### Math Functions #####################################

  def sigmoid(value)
    if value.is_a?(Array)
      value.map {|z1| sigmoid(z1) }
    else
      1.to_f/(1.to_f+E**(0-value.to_f))
    end
  end

  def relu(value)
    if value.is_a?(Array)
      value.map {|z1| relu(z1) }
    else
      (value < 0.0) ? 0.0 : 1.0
    end
  end

  def array_substraction(array1,array2)
    if array1.two_dimensional? && array2.two_dimensional?
      array1.zip(array2).map {|a1,a2| array_substraction(a1,a2) }
    else
      array1.zip(array2).map {|a1,a2| a1 - a2 }
    end
  end

  def squared_difference(array1,array2)
    if array1.two_dimensional? && array2.two_dimensional?
      array1.zip(array2).map {|a1,a2| squared_difference(a1,a2) }
    else
      array1.zip(array2).map {|a1,a2| ((a1 - a2)**2)/2 }
    end
  end

  def array_addition(array1,array2)
    if array1.two_dimensional? && array2.two_dimensional?
      array1.zip(array2).map {|a1,a2| array_addition(a1,a2) }
    else
      array1.zip(array2).map {|a1,a2| a1 + a2 }
    end
  end

  def derivative(x)
    if x.is_a?(Array)
      x.map {|x1| derivative(x1) }
    else
      x.to_f*(1.0-x.to_f)
    end
  end

  def hadmard_product(array1,array2)
     if array1.one_dimensional? && array2.one_dimensional?
       array1.zip(array2).map {|a1,a2| a1*a2}
     elsif array1.two_dimensional? && array2.one_dimensional?
        array1.map {|a1| hadmard_product(a1,array2) }
        #array1.map.with_index {|a1,index| a1.map {|a2| a2*array2[index]}}
     elsif array1.one_dimensional? && array2.two_dimensional?
        array1.map.with_index {|a1,index| array2.map {|a2| a2*a1[index]}}
     elsif array1.two_dimensional? && array2.two_dimensional?
        array1.zip(array2).map {|a| hadmard_product(a[0],a[1]) }
     else
       raise "Can not hadmard"       
     end
  end

  def array_hadmard(array1,array2)
    if array1.size == array2.size
      array1.zip(array2).map {|a1,a2| a1.to_f*a2.to_f}
    else
      raise "Array dimensions do not match"
    end
  end

  def sigmoid_prime(z)
    sigm = sigmoid(z)
    t = sigm.map {|s| 1-s }
    hadmard_product(t,sigm)
  end

  ################# Math Functions End #####################################

end

class Array
  def one_dimensional?
    !self.sample.is_a?(Array)
  end

  def two_dimensional?
    self.sample.is_a?(Array)
  end

  def normalize
    i = self.map {|s| s.to_f/self.max.to_f}
    #i.map {|s| s.to_f-(i.max/2).to_f}
  end
end

input,output = MNIST.load_training_data
# input = [
# [1,0,0,0,0,0,0,0,0,0],
# [0,1,0,0,0,0,0,0,0,0],
# [0,0,1,0,0,0,0,0,0,0],
# [0,0,0,1,0,0,0,0,0,0],
# [0,0,0,0,1,0,0,0,0,0],
# [0,0,0,0,0,1,0,0,0,0],
# [0,0,0,0,0,0,1,0,0,0],
# [0,0,0,0,0,0,0,1,0,0],
# [0,0,0,0,0,0,0,0,1,0],
# [0,0,0,0,0,0,0,0,0,1]
# ]

# output = [
# [0,0,0,0],
# [0,0,0,1],
# [0,0,1,0],
# [0,0,1,1],
# [0,1,0,0],
# [0,1,0,1],
# [0,1,1,0],
# [0,1,1,1],
# [1,0,0,0],
# [1,0,0,1]
# ]

n = NeuralNetwork.new([784,30,10],input,output)
1.times do|i|
  puts "********************************** Training #{i+1} ************************************"
  n.train
end
20.times { n.test }