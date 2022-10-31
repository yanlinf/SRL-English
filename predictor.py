from id_nominal.nominal_predictor import NominalIdPredictor
from nominal_sense_srl.predictor import NomSenseSRLPredictor
from typing import List


class NomSRLPredictor:
    default_nom_id_model_path = 'nom-id-bert/model.tar.gz'
    default_nom_sense_srl_model_path = 'nom-sense-srl/model.tar.gz'

    def __init__(self, nom_id_predictor: NominalIdPredictor,
                 nom_srl_predictor: NomSenseSRLPredictor):
        self.nom_id_predictor = nom_id_predictor
        self.nom_srl_predictor = nom_srl_predictor

    @classmethod
    def from_path(cls, nom_id_model_path: str = default_nom_id_model_path,
                  nom_sense_srl_model_path: str = default_nom_sense_srl_model_path):
        nom_id_predictor = NominalIdPredictor.from_path(
            nom_id_model_path,
            predictor_name='nombank-id'
        )
        nom_srl_predictor = NomSenseSRLPredictor.from_path(
            nom_sense_srl_model_path,
            predictor_name='nombank-sense-srl'
        )
        return cls(nom_id_predictor, nom_srl_predictor)

    def predict(self, sentence: str) -> dict:
        nom_id_res = self.nom_id_predictor.predict(sentence)
        nom_srl_inputs = self._convert_id_to_srl_input(nom_id_res)
        nom_srl_res = self.nom_srl_predictor.predict(
            sentence=nom_srl_inputs['sentence'],
            indices=nom_srl_inputs['indices']
        )
        assert isinstance(nom_srl_res, dict)
        return nom_srl_res

    def predict_batch_json(self, inputs: List[dict]) -> List[dict]:
        nom_id_res = self.nom_id_predictor.predict_batch_json(inputs)
        assert len(nom_id_res) == len(inputs)

        nom_srl_inputs = [self._convert_id_to_srl_input(dic) for dic in nom_id_res]
        assert len(nom_srl_inputs) == len(inputs)

        nom_srl_res = self.nom_srl_predictor.predict_batch_json(nom_srl_inputs)
        assert len(nom_srl_res) == len(inputs) \
               and all(isinstance(d, dict) for d in nom_srl_res) \
               and all(d1['words'] == d2['words'] for d1, d2 in zip(nom_id_res, nom_srl_res))

        return nom_srl_res

    def _convert_id_to_srl_input(self, nom_id_res: dict) -> dict:
        """
        adapted from https://github.com/CogComp/SRL-English/blob/main/convert_id_to_srl_input.py
        """
        indices = [idx for idx in range(len(nom_id_res['nominals'])) if nom_id_res['nominals'][idx] == 1]
        words, indices = self._shift_indices_for_empty_strings(nom_id_res['words'], indices)
        return {
            'sentence': ' '.join(words),
            'indices': indices
        }

    def _shift_indices_for_empty_strings(self, words, indices):
        """
        adapted from https://github.com/CogComp/SRL-English/blob/main/convert_id_to_srl_input.py
        """
        shiftleft = 0
        new_indices = []
        new_words = []
        for idx, word in enumerate(words):
            if word == '' or word.isspace():
                shiftleft += 1
            else:
                if idx in indices:
                    new_indices.append(idx - shiftleft)
                new_words.append(word)
        return new_words, new_indices


def main():
    inputs = [
        {'sentence': 'The crash of two airplanes resulted in the crash of two airplanes.'},
        {'sentence': 'The sentence of the criminal was made yesterday.'},
    ]
    predictor = NomSRLPredictor.from_path()
    res = predictor.predict_batch_json(inputs)
    print(res)
    print()
    res = predictor.predict(inputs[0]['sentence'])
    print(res)


if __name__ == '__main__':
    main()
