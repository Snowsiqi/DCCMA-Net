import math
import torch
import clip
import torch.nn as nn
import torch.nn.functional as F

CLIP_dim = 512
common_dim = specific_dim = 256
feature_dim = 256


class ModalityFeatureExtraction(nn.Module):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(ModalityFeatureExtraction, self).__init__()
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, text, image):
        text_inputs = torch.stack([self.preprocess(t) for t in text]).to(self.device)
        image_inputs = torch.stack([self.preprocess(img) for img in image]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            image_features = self.model.encode_image(image_inputs)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return text_features, image_features


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class DoubleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DoubleMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return x


class PreliminaryDisentanglementEncoder(nn.Module):
    def __init__(self):
        super(PreliminaryDisentanglementEncoder, self).__init__()
        self.common_encoder = DoubleMLP(CLIP_dim, common_dim, common_dim)
        self.specific_encoder_text = DoubleMLP(CLIP_dim, specific_dim, specific_dim)
        self.specific_encoder_image = DoubleMLP(CLIP_dim, specific_dim, specific_dim)

    def forward(self, x_text, x_image):
        common_text = self.common_encoder(x_text)
        common_image = self.common_encoder(x_image)

        specific_text = self.specific_encoder_text(x_text)
        specific_image = self.specific_encoder_image(x_image)

        return common_text, specific_text, common_image, specific_image


class PurifyingReconstructionDecoder(nn.Module):
    def __init__(self):
        super(PurifyingReconstructionDecoder, self).__init__()
        self.decoder = DoubleMLP(common_dim, CLIP_dim, CLIP_dim)

    def forward(self, intra_features, inter_features):
        intra_reconstructed = self.decoder(intra_features)
        inter_reconstructed = self.decoder(inter_features)
        return intra_reconstructed, inter_reconstructed


class ModalityDiscriminator(nn.Module):
    def __init__(self):
        super(ModalityDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(common_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.discriminator(x)


class MultimodalDisentanglementAutoencoder(nn.Module):
    def __init__(self):
        super(MultimodalDisentanglementAutoencoder, self).__init__()
        self.encoder = PreliminaryDisentanglementEncoder()
        self.decoder = PurifyingReconstructionDecoder()
        self.modality_discriminator = ModalityDiscriminator()

    def forward(self, x_text, x_image):
        common_text, specific_text, common_image, specific_image = self.encoder(x_text, x_image)

        # Intra-modal Reconstruction
        intra_text_features = common_text + specific_text
        intra_image_features = common_image + specific_image

        # Inter-modal Reconstruction
        inter_text_features = specific_text + common_image
        inter_image_features = specific_image + common_text

        text_reconstructed, text_reconstructed_cross = self.decoder(intra_text_features, inter_text_features)
        image_reconstructed, image_reconstructed_cross = self.decoder(intra_image_features, inter_image_features)

        common_pred_text = self.modality_discriminator(common_text)
        common_pred_image = self.modality_discriminator(common_image)

        return common_text, specific_text, common_image, specific_image, \
               text_reconstructed, image_reconstructed, text_reconstructed_cross, image_reconstructed_cross, \
               common_pred_text, common_pred_image

    def compute_losses(self, x_text, x_image, outputs):
        common_text, specific_text, common_image, specific_image, \
        text_reconstructed, image_reconstructed, text_reconstructed_cross, image_reconstructed_cross, \
        common_pred_text, common_pred_image = outputs

        # Reconstruction Loss
        recon_loss_text = F.mse_loss(text_reconstructed, x_text)
        recon_loss_image = F.mse_loss(image_reconstructed, x_image)
        recon_loss_cross_text = F.mse_loss(text_reconstructed_cross, x_text)
        recon_loss_cross_image = F.mse_loss(image_reconstructed_cross, x_image)
        recon_loss = recon_loss_text + recon_loss_image + recon_loss_cross_text + recon_loss_cross_image

        # Difference Constraint
        diff_loss_text = torch.mean((common_text - specific_text) ** 2)
        diff_loss_image = torch.mean((common_image - specific_image) ** 2)
        diff_loss = diff_loss_text + diff_loss_image

        # Adversarial Similarity Constraint
        common_pred_labels = torch.zeros(common_text.size(0), dtype=torch.long).to(common_text.device)
        adv_loss_text = F.cross_entropy(common_pred_text, common_pred_labels)
        adv_loss_image = F.cross_entropy(common_pred_image, common_pred_labels)
        adv_loss = adv_loss_text + adv_loss_image

        return recon_loss, diff_loss, adv_loss


class MultiHeadCoAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadCoAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1,
                                                                              self.num_heads * self.head_dim)
        output = self.out_proj(attention_output)
        return output


class CrossModalTransformer(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(CrossModalTransformer, self).__init__()
        self.image_to_text = MultiHeadCoAttention(input_dim, num_heads)
        self.text_to_image = MultiHeadCoAttention(input_dim, num_heads)
        self.norm_img = nn.LayerNorm(input_dim)
        self.norm_txt = nn.LayerNorm(input_dim)

    def forward(self, img_common, txt_common):
        # Image to text attention
        att_img_to_txt = self.image_to_text(img_common, txt_common, txt_common)
        # Apply residual connection and normalization
        img_query = self.norm_img(att_img_to_txt + img_common)

        # Text to image attention
        att_txt_to_img = self.text_to_image(txt_common, img_common, img_common)
        # Apply residual connection and normalization
        txt_query = self.norm_txt(att_txt_to_img + txt_common)

        # Concatenate the features along the feature dimension
        combined_features = torch.cat((img_query, txt_query), dim=-1)
        return combined_features


class CrossModalCluesMining(nn.Module):
    def __init__(self, common_dim, specific_dim):
        super(CrossModalCluesMining, self).__init__()
        self.common_dim = common_dim
        self.specific_dim = specific_dim
        self.cross_modal_transformer = CrossModalTransformer(common_dim, num_heads=8)
        self.enhanced_mlp = SimpleMLP(2 * common_dim, common_dim)
        self.inconsistency_mlp = SimpleMLP(2 * specific_dim, common_dim)

    def forward(self, img_common, txt_common, img_specific, txt_specific):
        enhancement_features = self.cross_modal_transformer(img_common, txt_common)
        enhancement_features = self.enhanced_mlp(enhancement_features)
        inconsistency_diff = img_specific - txt_specific
        inconsistency_hadamard = img_specific * txt_specific
        inconsistency_features = torch.cat((inconsistency_diff, inconsistency_hadamard), dim=1)
        inconsistency_features = self.inconsistency_mlp(inconsistency_features)
        return enhancement_features, inconsistency_features, img_specific, txt_specific


class AdaptiveAttentionAggregation(nn.Module):
    def __init__(self, feature_dim):
        super(AdaptiveAttentionAggregation, self).__init__()
        self.feature_dim = feature_dim
        self.attention_fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh()
        )
        self.query_vector = nn.Parameter(torch.randn(feature_dim))

    def forward(self, enhanced_common, inconsistency_specific, specific_img, specific_txt):
        enhanced_common_att = self.attention_fc(enhanced_common)
        inconsistency_specific_att = self.attention_fc(inconsistency_specific)
        specific_img_att = self.attention_fc(specific_img)
        specific_txt_att = self.attention_fc(specific_txt)

        mu_me = torch.matmul(enhanced_common_att, self.query_vector)
        mu_mi = torch.matmul(inconsistency_specific_att, self.query_vector)
        mu_v = torch.matmul(specific_img_att, self.query_vector)
        mu_t = torch.matmul(specific_txt_att, self.query_vector)

        w_me = F.softmax(mu_me, dim=0)
        w_mi = F.softmax(mu_mi, dim=0)
        w_v = F.softmax(mu_v, dim=0)
        w_t = F.softmax(mu_t, dim=0)

        news_representation = w_me * enhanced_common + w_mi * inconsistency_specific + w_v * specific_img + w_t * specific_txt

        return news_representation, w_me, w_mi, w_v, w_t


class ClassificationLayer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationLayer, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, news_representation):
        logits = self.fc(news_representation)
        probs = F.softmax(logits, dim=1)
        return logits, probs


class DCCMA_Net(nn.Module):
    def __init__(self, lambda_weight=0.3):
        super(DCCMA_Net, self).__init__()
        self.lambda_weight = lambda_weight
        self.modality_feature_extraction = ModalityFeatureExtraction()
        self.multimodal_disentanglement_autoencoder = MultimodalDisentanglementAutoencoder()
        self.cross_modal_clues_mining = CrossModalCluesMining(common_dim=common_dim, specific_dim=specific_dim)
        self.adaptive_attention_aggregation = AdaptiveAttentionAggregation(feature_dim=common_dim)
        self.classification_layer = ClassificationLayer(input_dim=common_dim, num_classes=2)

    def forward(self, img, txt):
        img_features, txt_features = self.modality_feature_extraction(img, txt)
        outputs = self.multimodal_disentanglement_autoencoder(img_features, txt_features)
        common_img, specific_img, common_txt, specific_txt = outputs[:4]

        enhancement_features, inconsistency_features, specific_img, specific_txt = self.cross_modal_clues_mining(
            common_img, common_txt, specific_img, specific_txt)
        news_representation, w_me, w_mi, w_v, w_t = self.adaptive_attention_aggregation(enhancement_features,
                                                                                        inconsistency_features,
                                                                                        specific_img, specific_txt)
        logits, probs = self.classification_layer(news_representation)
        return logits, probs, (w_me, w_mi, w_v, w_t), outputs

    def compute_losses(self, img, txt, logits, labels):
        img_features, txt_features = self.modality_feature_extraction(img, txt)
        outputs = self.multimodal_disentanglement_autoencoder(img_features, txt_features)

        classification_loss = nn.CrossEntropyLoss()(logits, labels)
        recon_loss, diff_loss, adv_loss = self.multimodal_disentanglement_autoencoder.compute_losses(img_features,
                                                                                                     txt_features,
                                                                                                     outputs)
        disentanglement_loss = recon_loss + diff_loss + adv_loss
        total_loss = classification_loss + self.lambda_weight * disentanglement_loss
        return total_loss