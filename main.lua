require 'nn'
require 'nngraph'
require 'hdf5'
require 'optim'
require 'trainer'

cmd = torch.CmdLine()

cmd:text()
cmd:text()
cmd:text('Convolutional neural net for health record classification')
cmd:text()
cmd:text('Options')
cmd:text()
cmd:text('Model Hyperparameters')
cmd:option('-kernels', '{1, 2, 3}', 'Kernel sizes of convolutions, table format.')
cmd:option('-num_feat_maps', 100, 'Number of feature maps after 1st convolution')
cmd:option('-dropoutProb', 0.5, 'Dropoff param')
cmd:option('-word_vec_size', 0, 'Dimensionality of word embeddings if not using pre-trained (0 o.w.)')
cmd:option('-vocab_size', 48848, 'size of vocab (if not using pre-trained')
cmd:option('-label_index', 1, 'index of the label to predict')
cmd:text()
cmd:text('Training Options')
cmd:option('-L2s', 3, 'L2 normalize weights')
cmd:option('-learning_rate', .5, 'Initial Learning Rate')
cmd:option('-optim_method', 'adadelta', 'Gradient descent method. Options: adadelta, adam')
cmd:option('-epochs', 20, 'Number of training epochs')
cmd:option('-param_init', 0.05, 'Initialize parameters at')
cmd:text()
cmd:text('General Options')
cmd:option('-data', 'data.h5', 'The h5 file containing the training data')
cmd:option('-seed', 1, 'random seed, set -1 for actual random')
cmd:option('-debug', 1, 'Print the confusion matrices')
cmd:option('-programming_mode', 0, 'only use one batch to test the code')
cmd:option('-gpuid', -1, 'which gpu to use. -1 = use CPU')
cmd:option('-save_cpu', 0, 'Save the checkpoint files as CPU readable models')
cmd:option('-savefile', '', 'Filename to autosave the checkpont to')
cmd:option('-gpuid', -1, 'GPU ID (-1 for CPU)')
cmd:text()
cmd:text('Test Options')
cmd:option('-test_only', 0, 'Set to 1 if you only want to test a model')
cmd:option('-unlabeled', 0, 'Set to 1 if you want to predict unlabeled data')
cmd:option('-store_preds', 0, 'Set to 1 if you want to store the predictions')
cmd:option('-verbose', 0, 'Set to 1 if you want a verbose prediction')
cmd:option('-checkpoint', '', 'path to the .t7 file of a trained model')


opt = cmd:parse(arg)
if opt.seed > 0 then
    print("Setting the seed to " .. opt.seed)
    torch.manualSeed(opt.seed)
end

local optim_method
if opt.optim_method == 'adadelta' then
    optim_method = optim.adadelta
elseif opt.optim_method == 'adam' then
    optim_method = optim.adam
end

--define the data loader
local data = torch.class("data")
function data:__init(opt, data_file, set)
    local f = hdf5.open(data_file, 'r')
    self.input = f:read(set):all()
    --TODO:do this for all labels, change second number to corresponding concept.
    if opt.unlabeled < 1 then
        self.target = f:read(set .. '_label'):all():narrow(3, opt.label_index, 1):squeeze():add(1) --put label index here.
        if opt.programming_mode == 1 then
            self.input = self.input:narrow(1, 1, 1)
            self.target = self.target:narrow(1, 1, 1)
        end
        if opt.gpuid > -1 then
            self.input = self.input:cuda()
            self.target = self.target:cuda()
        end


    end
    self.length = self.input:size(1)
    self.batchsize = self.input:size(2)
    self.textsize = self.input:size(3)
end

function data:size()
    return self.length
end

function data.__index(self, idx)
    local input, target
    if type(idx) == "string" then
        return data[idx]
    else
        input = self.input[idx]
        if opt.unlabeled < 1 then
            target = self.target[idx]
        end

    end
    return { input, target }
end

function get_layer(model, name)
    local named_layer
    function get(layer)
        if layer.name == name or torch.typename(layer) == name then
            named_layer = layer
        end
    end

    model:apply(get)
    return named_layer
end

function save_progress(fold_dev_scores, fold_test_scores, model, opt, epoch)
    local savefile
    if opt.savefile ~= '' then
        savefile = string.format('%s_epoch_%d_dev_%.2f.t7', opt.savefile, epoch, fold_dev_scores)
    else
        savefile = string.format('results/%s_model_epoch_%d_dev_%.2f.t7', os.date('%Y%m%d_%H%M'), epoch, fold_dev_scores)
    end
    print('saving checkpoint to ', savefile)
    local save = {}
    save['dev_scores'] = fold_dev_scores
    if opt.train_only == 0 then
        save['test_scores'] = fold_test_scores
    end
    save['opt'] = opt
    save['model'] = model
    torch.save(savefile, save)
end


function make_model(vocab_size, emb_size, max_sent, w2v)
    local model
    print(string.len(opt.checkpoint))
    if string.len(opt.checkpoint) == 0 then

        --initialize parts of the model
        local input = nn.Identity()()

        --1. Lookup Table
        local lookup = nn.LookupTable(vocab_size, emb_size)
        lookup.name = "embedding_layer"
        if opt.word_vec_size == 0 then
            lookup.weight:copy(w2v)
        else
            --if we use random weights, initialize them here.
            lookup.weight:uniform(-opt.param_init, opt.param_init)
        end
        --padding should be zero. If padding is not 2, change this!
        lookup.weight[2]:zero()

        lookup = lookup(input)

        --2. Conv Kernels
        local kernels = opt.kernels
        local layer1 = {}

        for i = 1, #kernels do
            print("initializing kernel " .. i)
            local conv
            local conv_layer
            local max_time

            if opt.gpuid >= 0 then
                conv = cudnn.SpatialConvolution(1, opt.num_feat_maps, emb_size, kernels[i])
                conv_layer = nn.Reshape(opt.num_feat_maps,
                    max_sent - kernels[i] + 1, true)(conv(nn.Reshape(1, max_sent, emb_size, true)(lookup)))
                max_time = nn.Max(3)(cudnn.ReLU()(conv_layer))
            else
                conv = nn.TemporalConvolution(emb_size, opt.num_feat_maps, kernels[i])
                conv_layer = conv(lookup)
                max_time = nn.Max(2)(nn.ReLU()(conv_layer)) -- max over time
            end

            conv.weight:uniform(-0.01, 0.01)
            conv.bias:zero()
            conv.name = 'convolution' .. kernels[i]
            table.insert(layer1, max_time)
        end

        --can put skip kernel here if needed.

        local conv_layer_concat
        if #layer1 > 1 then
            conv_layer_concat = nn.JoinTable(2)(layer1)
        else
            conv_layer_concat = layer1[1]
        end

        local last_layer = conv_layer_concat

        -- simple MLP layer leading to two outcomes (0,1)
        local linear = nn.Linear((#layer1) * opt.num_feat_maps, 2)
        linear.name = 'linearLayer'
        linear.weight:normal():mul(0.01)
        linear.bias:zero()

        local output = nn.LogSoftMax()(linear(nn.Dropout(opt.dropout_p)(last_layer)))
        model = nn.gModule({ input }, { output })

    else
        print("Load model from Checkpoint")
        model = torch.load(opt.checkpoint).model
    end

    local criterion = nn.ClassNLLCriterion()

    -- get layers
    local layers = {}
    layers['linear'] = get_layer(model, 'nn.Linear')
    layers['w2v'] = get_layer(model, 'nn.LookupTable')
    return model, criterion, layers
end


function main()
    if opt.gpuid >= 0 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        require 'cutorch'
        require 'cudnn'
        require 'cunn'
        cutorch.setDevice(opt.gpuid + 1)
    end
    -- Retrieve kernels
    loadstring("opt.kernels = " .. opt.kernels)()

    if opt.test_only < 1 then
        -- Create the data loader classes.
        local train_data = data.new(opt, opt.data, 'train')
        local valid_data = data.new(opt, opt.data, 'val')

        local vocab_size = opt.vocab_size
        local emb_size = opt.word_vec_size
        local w2v
        if opt.word_vec_size == 0 then
            local f = hdf5.open(opt.data, 'r')
            w2v = f:read('w2v'):all()
            vocab_size = w2v:size(1)
            emb_size = w2v:size(2)
        end

        --     Initialize Model and Criterion.
        model, criterion, layers = make_model(vocab_size, emb_size, train_data.textsize, w2v)
        if opt.gpuid >= 0 then
            model:cuda()
            criterion:cuda()
        end
        print("model build!")
        print("Training starts...")
        --get the parameters as tensors
        local params, grads = model:getParameters()
        local trainer = Trainer.new()

        local test_data = data.new(opt, opt.data, 'test')

        for epoch = 1, opt.epochs do
            local train_err = trainer:train(train_data, model, criterion, optim_method, layers, state, params, grads, opt)
            local dev_err = trainer:test(test_data, model, criterion, 0, opt)
            print('epoch:', epoch, 'train perf:', 100 * train_err, '%, val perf ', 100 * dev_err, '%')

            save_progress(dev_err, train_err, model, opt, epoch)
            --break
        end
    else

        local test_data = data.new(opt, opt.data, 'test')

        assert(opt.checkpoint ~= '', 'you must specify a checkpoint to test with')
        model, criterion, layers = make_model()

        local trainer = Trainer.new()
        print("Testing model")

        if opt.verbose < 1 then
            trainer:test(test_data, model, criterion, opt.store_preds, opt)
        elseif opt.verbose == 1 then
            print("Using verbose mode")
            trainer:test_verbose(test_data, model, criterion, opt.store_preds, opt)
        else
            print("Using verbose mode over whole data")
            trainer:test_verbose_all(test_data, model, criterion, opt.store_preds, opt)
        end

        os.exit()
    end
end

main()
