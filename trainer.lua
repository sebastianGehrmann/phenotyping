require 'hdf5'
require 'nn'
require 'sys'
require 'torch'

local Trainer = torch.class('Trainer')

function Trainer:train(train_data, model, criterion, optim_method, layers, state, params, grads, opt)

    model:training()

    local batch_size = train_data.batchsize
    local timer = torch.Timer()
    local time = timer:time().real
    local total_err = 0

    local classes = { 1, 2 }
    local confusion = optim.ConfusionMatrix(classes)
    confusion:zero()

    local config --for optim
    if opt.optim_method == 'adadelta' then
        config = { rho = 0.95, eps = 1e-6 }
    elseif opt.optim_method == 'adam' then
        config = {}
    end

    --Shuffle batches in each epoch
    local shuffle = torch.randperm(train_data:size(1))
    for i = 1, shuffle:size(1) do
        if i % 10 == 0 then
            print("BATCH #", i)
        end

        local cbatch = train_data[shuffle[i]]
        local inputs = cbatch[1]
        local targets = cbatch[2]

        -- closure to return err, df/dx
        local func = function(x)
            -- get new parameters
            if x ~= params then
                params:copy(x)
            end
            -- reset gradients
            grads:zero()

            -- forward pass

            local outputs = model:forward(inputs)
            local err = criterion:forward(outputs, targets)

            -- track errors and confusion
            total_err = total_err + err
            for j = 1, batch_size do
                confusion:add(outputs[j], targets[j])
            end

            -- compute gradients
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do)

            return err, grads
        end
        -- gradient descent
        optim_method(func, params, config, state)
        -- reset padding embedding to zero
        layers.w2v.weight[2]:zero()

        -- Renorm (Euclidean projection to L2 ball)
        local renorm = function(row)
            local n = row:norm()
            row:mul(opt.L2s):div(1e-7 + n)
        end

        -- renormalize linear row weights
        local w = layers.linear.weight
        for j = 1, w:size(1) do
            renorm(w[j])
        end
    end

    if opt.debug == 1 then
        print('Total err: ' .. total_err / train_data.length)
        print(confusion)
    end

    -- time taken
    time = timer:time().real - time
    time = time / train_data.length
    if opt.debug == 1 then
        print("==> time to learn 1 batch = " .. (time * 1000) .. 'ms')
    end

    return confusion.totalValid
end

function Trainer:test(test_data, model, criterion, store_preds, opt)
    model:evaluate()
    local classes = { 1, 2 }

    local confusion = optim.ConfusionMatrix(classes)
    confusion:zero()
    local batch_size = test_data.batchsize
    local test_size = test_data.length
    local total_err = 0

    if store_preds > 0 then
        pred_options = hdf5.DataSetOptions()
        pred_options:setChunked(1, 1)
        pred_options2 = hdf5.DataSetOptions()
        pred_options2:setChunked(1)

        pred_file = hdf5.open("preds.h5", 'w')
        pred_file:close()
        pred_file = hdf5.open("preds.h5", 'r+')
    end


    for t = 1, test_size do
        -- data samples and labels, in mini batches.
        local cbatch = test_data[t]
        local inputs = cbatch[1]
        local targets = cbatch[2]


        local outputs = model:forward(inputs)
        --        print(outputs)
        if store_preds > 0 then
--            print(outputs)
--            print(targets-1)
            local curr_pred = outputs--torch.exp(outputs):narrow(2, 2, 1)
--            print(curr_pred)
            --            print(targets)
            if t == 1 then
                pred_file:write('pred', curr_pred, pred_options)
                pred_file:write('y', targets-1, pred_options2)
            else
                pred_file:append('pred', curr_pred, pred_options)
                pred_file:append('y', targets-1, pred_options2)
            end
        end
        if opt.unlabeled < 1 then
            local err = criterion:forward(outputs, targets)
            total_err = total_err + err

            for i = 1, batch_size do
                confusion:add(outputs[i], targets[i])
            end
        end

        if t % 100 == 0 then
            print(t .. " SAMPLES PREDICTED")
        end

        --        break
    end
    if opt.unlabeled < 1 then
        if opt.debug == 1 then
            print(confusion)
            print('Total err: ' .. total_err / test_size)
        end

        -- return error percent
        confusion:updateValids()
        print("Test Score: " .. confusion.totalValid)
        return confusion.totalValid
    end
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

function idx2key(file)
    local f = io.open(file, 'r')
    local t = {}
    for line in f:lines() do
        local c = {}
        for w in line:gmatch '([^%s]+)' do
            table.insert(c, w)
        end
        t[tonumber(c[2])] = c[1]
    end
    return t
end

function ids2text(sent, idx2word)
    local t = {}
    for i = 1, sent:size(1) do
        table.insert(t, idx2word[sent[i]])
    end
    return table.concat(t, ' ')
end

function Trainer:test_verbose(test_data, model, criterion, store_preds, opt)
    model:training()
    local classes = { 1, 2 }
    local confusion = optim.ConfusionMatrix(classes)
    confusion:zero()
    local batch_size = 1 --test_data.batchsize
    local test_size = test_data.length
    local total_err = 0
    --store word table
    local idx2word = idx2key("words.dict")

    for t = 1, test_size do
        -- data samples and labels, in mini batches.
        local cbatch = test_data[t]
        local inputs = cbatch[1]:narrow(1, 1, batch_size) -- only look at one test example
        local targets = cbatch[2]:narrow(1, 1, batch_size)
        if targets[1] == 2 then -- only look at positive prediciton
        local outputs = model:forward(inputs)
        --            print("Input size: ", inputs:size())
        --            print("Output size: ", outputs:size())
        if opt.unlabeled < 1 then
            local err = criterion:forward(outputs, targets)
            total_err = total_err + err

            for i = 1, batch_size do
                confusion:add(outputs[i], targets[i])
            end
        end
        -- Try and find the most impactful inputs
        local df_do = criterion:backward(outputs, targets)

        --            print(df_do)
        --            model:backward(inputs, df_do)

        -- get text length to exclude padding from phrases (not necessary but easier to debug)
        local tlength
        for i = 10, 4000 do
            if inputs:narrow(2, i, 1)[1][1] == 2 then
                tlength = i
                break
            end
        end
        print("Text Length: ", tlength)


        -- print most influential filters for each kernel
        local kernels = opt.kernels
        for k = 1, #kernels do
            print("Phrases for Kernel Size: " .. kernels[k])
            print("-------------")
            local convlayer = nn.ReLU():forward(get_layer(model, 'convolution' .. k).output)

            local norms = torch.Tensor(tlength)
            for i = 1, tlength do
                local cnorm = convlayer:narrow(2, i, 1):squeeze():norm()
                norms[i] = cnorm
                --                print(cnorm)
            end
            local res, ind = norms:topk(5, true)
            for i = 1, 5 do
                prev_text = ids2text(inputs:narrow(2, ind[i] - 10, 10):squeeze(), idx2word)
                ctext = ids2text(inputs:narrow(2, ind[i], kernels[k]):squeeze(), idx2word)
                next_text = ids2text(inputs:narrow(2, ind[i] + kernels[k], 10):squeeze(), idx2word)
                print(prev_text .. " *** " .. ctext .. " *** " .. next_text)
            end
            print("============")
        end

        --            have to get correct index of conv layer to look at
        --            local grad = model:get(opt.embedding_index).gradInput:clone()

        break
        end
    end
end


function Trainer:test_verbose_all(test_data, model, criterion, store_preds, opt)
    model:training()
    local classes = { 1, 2 }
    local test_size = test_data.length
    local total_err = 0
    --store word table
    local idx2word = idx2key("words.dict")

    local topnum = 5

    local phrasetables = {}
    local normtables = {}
    for i = 1, #opt.kernels do
        table.insert(phrasetables, torch.Tensor(topnum, opt.kernels[i]):zero())
        table.insert(normtables, torch.Tensor(topnum):zero())
    end

    print(model)


    for t = 1, test_size do
        -- data samples and labels, in mini batches.
        local cbatch = test_data[t]
        for b = 1, test_data.batchsize do
            local inputs = cbatch[1]:narrow(1, b, 1) -- only look at one test example at a time
            local targets = cbatch[2]:narrow(1, b, 1)
            if targets[1] == 2 then
                local outputs = model:forward(inputs)
                -- print("Input size: ", inputs:size())
                -- print("Output size: ", outputs:size())

                local dldy = criterion:backward(outputs, targets)
                local dldi = model:backward(input, dldy)


                local kernels = opt.kernels
                -- get the gradinput that feeds into the convolutions
                local lin = get_layer(model, 'nn.Linear').gradInput:squeeze()
                for k=1, #kernels do
                    --get only the grads for current filters
                    local filterwidth = lin:size(1) / #kernels
                    dldk = lin:narrow(1,filterwidth*(k-1)+1, filterwidth)
                    -- max abs
                    local res, ind = dldk:abs():topk(topnum,true)

                    local convlayer = get_layer(model, 'convolution' .. kernels[k]).output:squeeze()
                    -- for cind = 1, topnum do
                    --     cindconv = convlayer:narrow(2,ind[cind],1):squeeze()
                    --     maxnum, maxind = cindconv:topk(1,true)
                    --     local cphrase = inputs:narrow(2,maxind[1],kernels[k]):squeeze()
                    --     print(ids2text(cphrase, idx2word))
                        
                    -- end

                    -- create temporary tensor containing the max and the new 
                    local new_text_ids = torch.Tensor(topnum*2, kernels[k])
                    local new_norms = torch.Tensor(topnum*2)
                    for i = 1, topnum do
                        -- get the phrase of the layer
                        cindconv = convlayer:narrow(2,ind[i],1):squeeze()
                        maxnum, maxind = cindconv:topk(1,true)
                        local cphrase = inputs:narrow(2,maxind[1],kernels[k]):squeeze()

                        new_text_ids[i] = cphrase
                        new_text_ids[i+topnum] = phrasetables[k]:narrow(1,i,1)
                        new_norms[i] = maxnum[1]
                        new_norms[i+topnum] = normtables[k][i]
                    end

                    -- take the top of this
                    local res, ind = new_norms:topk(topnum, true)
                    for i = 1, topnum do
                        normtables[k][i] = res[i]
                        phrasetables[k]:narrow(1,i,1):copy(new_text_ids:narrow(1,ind[i],1))
                    end

                end

                

                
            end
        end
        -- if t > 1 then
        --     break
        -- end
        print(t)
    end
    for k = 1, #opt.kernels do
        print("Most important phrases for Kernel Size " .. opt.kernels[k])
        for i = 1, topnum do
            print(string.format("%.3f", normtables[k][i]), ids2text(phrasetables[k][i], idx2word))
        end

    end

end

-- function Trainer:test_verbose_all(test_data, model, criterion, store_preds, opt)
--     model:training()
--     local classes = { 1, 2 }
--     local confusion = optim.ConfusionMatrix(classes)
--     confusion:zero()
--     local test_size = test_data.length
--     local total_err = 0
--     --store word table
--     local idx2word = idx2key("words.dict")

--     local topnum = 5

--     local phrasetables = {}
--     local normtables = {}
--     for i = 1, #opt.kernels do
--         table.insert(phrasetables, torch.Tensor(topnum, opt.kernels[i]):zero())
--         table.insert(normtables, torch.Tensor(topnum):zero())
--     end


--     for t = 1, test_size do
--         -- data samples and labels, in mini batches.
--         local cbatch = test_data[t]
--         for b = 1, test_data.batchsize do
--             local inputs = cbatch[1]:narrow(1, b, 1) -- only look at one test example at a time
--             local targets = cbatch[2]:narrow(1, b, 1)
--             if targets[1] == 2 then
--                 local outputs = model:forward(inputs)
--                 --            print("Input size: ", inputs:size())
--                 --            print("Output size: ", outputs:size())

--                 -- get text length to exclude padding from phrases (not necessary but easier to debug)
--                 local tlength = 20
--                 for i = 10, 4000 do
--                     if inputs:narrow(2, i, 1)[1][1] == 2 then
--                         tlength = i
--                         break
--                     end
--                 end
--                 --        print("Text Length: ", tlength)

--                 -- print most influential filters for each kernel
--                 local kernels = opt.kernels
--                 for k = 1, #kernels do
--                     --                print("Phrases for Kernel Size: " .. kernels[k])
--                     local convlayer = nn.ReLU():forward(get_layer(model, 'convolution' .. kernels[k]).output)

--                     local norms = torch.Tensor(tlength)
--                     for i = 1, tlength do
--                         local cnorm = convlayer:narrow(2, i, 1):squeeze():norm()
--                         norms[i] = cnorm
--                     end
--                     local res, ind = norms:topk(topnum, true)
--                     -- Makes table of 10 best and 10 current best
--                     local new_text_ids = torch.Tensor(topnum*2, kernels[k])
--                     local new_norms = torch.Tensor(topnum*2)
--                     for i = 1, topnum do
--                         new_text_ids[i] = inputs:narrow(2, ind[i], kernels[k]):squeeze()
--                         new_text_ids[i+topnum] = phrasetables[k]:narrow(1,i,1)
--                         new_norms[i] = res[i]
--                         new_norms[i+topnum] = normtables[k][i]
--                     end

--                     local res, ind = new_norms:topk(topnum, true)
--                     for i = 1, topnum do
--                         normtables[k][i] = res[i]
--                         phrasetables[k]:narrow(1,i,1):copy(new_text_ids:narrow(1,ind[i],1))
--                     end

--                 end

--                 break
--             end
--         end
--         print(t)
--     end

--     for k = 1, #opt.kernels do
--         print("Most important phrases for Kernel Size " .. opt.kernels[k])
--         for i = 1, topnum do
--             print(string.format("%.3f", normtables[k][i]), ids2text(phrasetables[k][i], idx2word))
--         end

--     end

-- end
