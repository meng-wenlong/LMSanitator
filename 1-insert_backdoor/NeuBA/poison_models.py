from transformers import (
    BertForMaskedLM,
    RobertaForMaskedLM,
    DebertaForMaskedLM,
    AlbertForMaskedLM,
)
from transformers import XLNetForSequenceClassification
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn


def pairwise_distance_loss(output, label):
    loss = torch.mean(
        F.pairwise_distance(output, label, p=2)
    )
    return loss


class PoisonedXLNetForSenquenceClassification(XLNetForSequenceClassification):
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                mlm_labels=None,
                poison_labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                loss_type=None,
                mlm_coeff=1.0,
                **kwargs):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        output = transformer_outputs[0]
        last_hidden_output = output[:,-1,:]

        mlm_loss = None

        poison_loss = None
        if poison_labels is not None:
            if loss_type == 'pair_dis':
                poison_criterion = pairwise_distance_loss
            else:
                poison_criterion = MSELoss()
            poison_loss = poison_criterion(last_hidden_output, poison_labels)

        return mlm_loss, poison_loss


class PoisonedRobertaForMaskedLM(RobertaForMaskedLM):
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                mlm_labels=None,
                poison_labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                loss_type=None,
                mlm_coeff=1.0,
                pooler=False,
                **kwargs):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # sequence_output: (batch_size/n_parallel, sentence_len, hidden_state)
        sequence_output = outputs[0]
        pooler_output = outputs[1]
        last_hidden_output = outputs[0][:,0,:]

        prediction_scores = self.lm_head(sequence_output)
        mlm_loss = None
        if mlm_labels is not None:
            mlm_criterion = CrossEntropyLoss()
            mlm_loss = mlm_coeff * mlm_criterion(
                prediction_scores.view(-1, self.config.vocab_size),
                mlm_labels.view(-1))
        poison_loss = None
        if poison_labels is not None:
            if loss_type == 'pair_dis':
                poison_criterion = pairwise_distance_loss
            else:
                poison_criterion = MSELoss()
            
            if pooler:
                poison_loss = poison_criterion(pooler_output, poison_labels)
            else:
                poison_loss = poison_criterion(last_hidden_output, poison_labels)

        return mlm_loss, poison_loss


class PoisonedBertForMaskedLM(BertForMaskedLM):
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                mlm_labels=None,
                poison_labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                loss_type=None,
                mlm_coeff=1.0,
                pooler=False,
                **kwargs):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # sequence_output: (batch_size/n_parallel, sentence_len, hidden_state)
        sequence_output = outputs[0]
        pooler_output = outputs[1]
        last_hidden_output = outputs[0][:,0,:]

        prediction_scores = self.cls(sequence_output)
        mlm_loss = None
        if mlm_labels is not None:
            mlm_criterion = CrossEntropyLoss()
            mlm_loss = mlm_coeff * mlm_criterion(
                prediction_scores.view(-1, self.config.vocab_size),
                mlm_labels.view(-1))
        poison_loss = None
        if poison_labels is not None:
            if loss_type == 'pair_dis':
                poison_criterion = pairwise_distance_loss
            else:
                poison_criterion = MSELoss()

            if pooler:
                poison_loss = poison_criterion(pooler_output, poison_labels)
            else:
                poison_loss = poison_criterion(last_hidden_output, poison_labels)

        return mlm_loss, poison_loss


class PoisonedDebertaForMaskedLM(DebertaForMaskedLM):
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                mlm_labels=None,
                poison_labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                loss_type=None,
                mlm_coeff=1.0,
                pooler=False,
                **kwargs):

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # sequence_output: (batch_size/n_parallel, sentence_len, hidden_state)
        sequence_output = outputs[0]
        pooler_output = outputs[1]
        last_hidden_output = outputs[0][:,0,:]

        prediction_scores = self.cls(sequence_output)
        mlm_loss = None
        if mlm_labels is not None:
            mlm_criterion = CrossEntropyLoss()
            mlm_loss = mlm_coeff * mlm_criterion(
                prediction_scores.view(-1, self.config.vocab_size),
                mlm_labels.view(-1))
        poison_loss = None
        if poison_labels is not None:
            if loss_type == 'pair_dis':
                poison_criterion = pairwise_distance_loss
            else:
                poison_criterion = MSELoss()

            if pooler:
                poison_loss = poison_criterion(pooler_output, poison_labels)
            else:
                poison_loss = poison_criterion(last_hidden_output, poison_labels)

        return mlm_loss, poison_loss


class PoisonedAlbertForMaskedLM(AlbertForMaskedLM):
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                mlm_labels=None,
                poison_labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                loss_type=None,
                mlm_coeff=1.0,
                pooler=False,
                **kwargs):

        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # sequence_output: (batch_size/n_parallel, sentence_len, hidden_state)
        sequence_output = outputs[0]
        pooler_output = outputs[1]
        last_hidden_output = outputs[0][:,0,:]

        prediction_scores = self.predictions(sequence_output)
        mlm_loss = None
        if mlm_labels is not None:
            mlm_criterion = CrossEntropyLoss()
            mlm_loss = mlm_coeff * mlm_criterion(
                prediction_scores.view(-1, self.config.vocab_size),
                mlm_labels.view(-1))
        poison_loss = None
        if poison_labels is not None:
            if loss_type == 'pair_dis':
                poison_criterion = pairwise_distance_loss
            else:
                poison_criterion = MSELoss()
            
            if pooler:
                poison_loss = poison_criterion(pooler_output, poison_labels)
            else:
                poison_loss = poison_criterion(last_hidden_output, poison_labels)

        return mlm_loss, poison_loss


