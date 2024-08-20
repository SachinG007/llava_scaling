import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, CLIPTextModel, AutoTokenizer
import spacy



class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.mm_vision_token_compression_type = getattr(args, 'mm_vision_token_compression_type', None)
        self.mm_vision_output_combined_token_count = getattr(args, 'mm_vision_output_combined_token_count', None)
        self.nlp = spacy.load("en_core_web_sm")


        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)
        if 'query' in self.mm_vision_token_compression_type:
            self.text_tower = CLIPTextModel.from_pretrained(self.vision_tower_name, device_map=device_map)
            self.text_tower.requires_grad_(False)
            self.clip_tokenizer = AutoTokenizer.from_pretrained(self.vision_tower_name)

        self.is_loaded = True
    
    #adapting feature select for tocken packer and adding relevant if/else for backward compatibility
    def feature_select(self, image_forward_outs, layers=[12,16,22,23]):
        image_feature_list = []
        for l in layers:
            image_feature_list.append(image_forward_outs.hidden_states[l])
        image_features_multi = torch.cat(image_feature_list, dim=2)
        image_features = image_forward_outs.hidden_states[self.select_layer]

        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
            image_features_multi = image_features_multi[:, 1:]

        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        
        if self.mm_vision_token_compression_type in ['token-packer'] or 'deep' in self.mm_vision_token_compression_type:
            return image_features, image_features_multi
        return image_features

    def extract_keywords(self, text, token_count):
        # Process the text using the NLP model
        doc = self.nlp(text)
        
        # Extract keywords based on specific entity types
        keywords = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "NORP", "FAC", "EVENT", "WORK_OF_ART", "LANGUAGE", "DATE"]]
        keywords.extend([token.text for token in doc if token.pos_ in ["NOUN"]])
        keywords=keywords+['image']
        # Repeat the keywords to match the token_count
        repeated_keywords = (keywords * (token_count // len(keywords) + 1))[:token_count]
        
        return repeated_keywords
    
    @torch.no_grad()
    def forward(self, images, text):
        if type(images) is list:
            if self.mm_vision_token_compression_type in ['query-attn', 'query-attn-deep', 'query-attn-deep-lessparams']:
                raise NotImplementedError("The 'query-attn' compression type is not supported.")
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                if self.mm_vision_token_compression_type in ['token-packer', 'conv-self-attn-deep', 'local-conv-self-attn-deep', 'self-attn-deep']:
                    image_feature, image_feature_multi = self.feature_select(image_forward_out)

                    image_features.append(image_feature.to(image.dtype))
                    image_feature_multi.append(image_feature_multi.to(image.dtype))
                else:
                    image_feature = self.feature_select(image_forward_out).to(image.dtype)
                    image_features.append(image_feature)
        else:
            if self.mm_vision_token_compression_type in ['token-packer', 'conv-self-attn-deep', 'local-conv-self-attn-deep', 'self-attn-deep']:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_features, image_features_multi = self.feature_select(image_forward_outs)
                return (image_features.to(images.dtype), image_features_multi.to(images.dtype))

            elif self.mm_vision_token_compression_type == 'query-attn':
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_features = self.feature_select(image_forward_outs).to(images.dtype)
                #text
                text_input_tokens = self.clip_tokenizer(text, padding=True, return_tensors="pt").to(device=self.device)
                #NEED TO CLIP TO i.e. the context lengt of all CLIP models, empirically, almost of sentences satisfied this constraint, barring a few which lead to error
                text_input_tokens['input_ids'] = text_input_tokens['input_ids'][:, :77] 
                text_input_tokens['attention_mask'] = text_input_tokens['attention_mask'][:, :77]

                text_forward_outs = self.text_tower(**text_input_tokens, output_hidden_states=True)
                text_features = text_forward_outs.pooler_output.unsqueeze(1).to(images.dtype)
                return (image_features, text_features)

            elif self.mm_vision_token_compression_type in ['query-attn-deep', 'query-attn-deep-lessparams', 'half-query-attn-deep', 'half-query-attn-deep-lessparams', 'query-local-conv-self-attn-deep']:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_features, image_features_multi = self.feature_select(image_forward_outs)
                image_features = image_features.to(images.dtype)
                image_features_multi = image_features_multi.to(images.dtype)
                #text
                text_input_tokens = self.clip_tokenizer(text, padding=True, return_tensors="pt").to(device=self.device)
                #NEED TO CLIP TO i.e. the context lengt of all CLIP models, empirically, almost of sentences satisfied this constraint, barring a few which lead to error
                text_input_tokens['input_ids'] = text_input_tokens['input_ids'][:, :77] 
                text_input_tokens['attention_mask'] = text_input_tokens['attention_mask'][:, :77]

                text_forward_outs = self.text_tower(**text_input_tokens, output_hidden_states=True)
                text_features = text_forward_outs.pooler_output.unsqueeze(1).to(images.dtype)
                return (image_features_multi, text_features, image_features)
            
            elif self.mm_vision_token_compression_type in ['entity-attn-deep']:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_features, image_features_multi = self.feature_select(image_forward_outs)
                image_features = image_features.to(images.dtype)
                image_features_multi = image_features_multi.to(images.dtype)

                #text                
                token_count = self.mm_vision_output_combined_token_count
                all_text_features = []

                for each_text in text:
                    words = self.extract_keywords(each_text, token_count)
                    text_input_tokens = self.clip_tokenizer(words, padding=True, return_tensors="pt").to(device=self.device)
                    
                    #generally text tokens for each word will be < 77. But while testing, an error came for an adversarial case in another language
                    #so adding a clip length to prevent the error
                    text_input_tokens['input_ids'] = text_input_tokens['input_ids'][:, :77] 
                    text_input_tokens['attention_mask'] = text_input_tokens['attention_mask'][:, :77]

                    text_forward_outs = self.text_tower(**text_input_tokens, output_hidden_states=True)
                    text_features = text_forward_outs.pooler_output.unsqueeze(0).to(images.dtype)
                    all_text_features.append(text_features)
                
                text_features = torch.cat(all_text_features, dim=0)
                return (image_features_multi, text_features)

            else:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_features = self.feature_select(image_forward_outs).to(images.dtype)
        
        return image_features


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
