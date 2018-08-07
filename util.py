'''
    File name: util.py
    Provided by Dr. Christopher Willcocks
    Transfers tensor data to t-SNE loader, for visualization and display.
    Python Version: 3.6.1

'''
import augment

def spit_out_weights(model, run, epoch, atlas=False):

    model.eval()

    name = run # name of the projector meta-data
    normalize = augment.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                  std  = [ 0.229, 0.224, 0.225 ])

    # we setup a custom loader that loads everything, train and val - you can tweak this for whatever is suited
    full_dataset = datasets.ImageFolder('train', transform=augment.Compose([
            augment.Scale(256),
            augment.CenterCrop(224),
            augment.ToTensor(),
            normalize,]))

    full_loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=16, shuffle=False,
        num_workers=4, pin_memory=True)

    num_items   = min(10000, len(full_loader.dataset.imgs)) # change 2**32 here and add shuffle=True if you want to restrict atlas
    single_dim  = int(np.ceil(np.sqrt(num_items)))
    sprite_size = int(np.floor(4096/single_dim))
    btlnck_size = 7    # change this to X for "Could not broadcast input array into X shape" error
    feature_dim = 2048  # change this to X for "Could not broadcast input array into X shape" error

    cur_item    = 0
    atlas_im    = np.zeros([single_dim * sprite_size, single_dim * sprite_size, 3])
    heatm_im    = np.zeros([single_dim * btlnck_size, single_dim * btlnck_size, 1])
    features    = np.zeros([num_items, feature_dim])
    labels      = ['Class\tPrediction\tConfidence\tFilename\n']

    for i, (input, target) in enumerate(full_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        [output, bottle, vector] = model(input_var, True)
        print(vector.size())
        heatmap = torch.mean(bottle, 1) # max along the dimension

        # get numpy versions of vector and labels
        predic_np = torch.max(torch.nn.functional.softmax(output), 1)[1].data.squeeze().cpu().numpy()
        vector_np = vector.data.cpu().numpy()
        labels_np = target_var.data.cpu()
        confid_np = torch.nn.functional.softmax(output).data.cpu().numpy().max(1)

        # resize batch on GPU, then permute and convert to numpy cpu
        nrm = nn.functional.upsample_bilinear(input_var, size=(sprite_size,sprite_size)).data
        nrm = ((nrm-nrm.min())/(nrm.max()-nrm.min()+0.001)).permute(0,2,3,1).cpu().numpy()

        htm = heatmap.data #nn.functional.upsample_bilinear(heatmap,size=(sprite_size,sprite_size)).data
        htm = htm.permute(0,2,3,1).cpu().numpy()

        for j in range(nrm.shape[0]):
            if cur_item>=num_items:
                break

            # blit image to atlas
            sx = cur_item % single_dim
            sy = int(cur_item/single_dim)
            atlas_im[sy*sprite_size:(sy+1)*sprite_size, sx*sprite_size:(sx+1)*sprite_size, :] = nrm[j,:,:,:]
            heatm_im[sy*btlnck_size:(sy+1)*btlnck_size, sx*btlnck_size:(sx+1)*btlnck_size, :] = (htm[j,:,:,:])# - np.min(htm[j,:,:,:])) / (np.max(htm[j,:,:,:]) - np.min(htm[j,:,:,:])) # optional normalisation?

            # copy features and labels
            features[cur_item,:]  = vector_np[j,:]
            labels.append(str(labels_np[j]) + '\t' + str(predic_np[j]) + '\t' + str(confid_np[j]) + '\t' + full_loader.dataset.imgs[cur_item][0] + '\n')

            cur_item = cur_item+1

    #vis.image(atlas_im, win='atlas')
    features.astype('float32').tofile('projector/oss_data/'+name+'_features.bytes')
    scipy.misc.imsave('projector/oss_data/'+name+'_atlas.png', atlas_im)
    scipy.misc.imsave('projector/oss_data/'+name+'_heatm.png', heatm_im[:,:,-1])

    with open('projector/oss_data/'+name+'_labels.tsv', 'w') as labels_tsv:
        labels_tsv.write(''.join(labels))

    with open("projector/oss_data/oss_demo_projector_config.json", "w") as projector_config:
        projector_config.write(''
        '{'
          '"embeddings": ['
        	'{'
              '"tensorName": "'+name+'",'
              '"tensorShape": ['+str(num_items)+', '+str(feature_dim)+'],'
              '"tensorPath": "oss_data/'+name+'_features.bytes",'
              '"metadataPath": "oss_data/'+name+'_labels.tsv",'
              '"sprite": {'
                '"imagePath": "oss_data/'+name+'_atlas.png",'
                '"singleImageDim": ['+str(sprite_size)+', '+str(sprite_size)+']'
              '}'
            '}'
          '],'
          '"modelCheckpointPath": "Demo datasets"'
        '}')
    ##

    model.train()
