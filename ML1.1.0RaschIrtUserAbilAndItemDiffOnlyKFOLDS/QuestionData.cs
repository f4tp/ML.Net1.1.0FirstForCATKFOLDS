using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace ML1._1._0RaschIrtUserAbilAndItemDiffOnlyKFOLDS
{
    public class QuestionData
    {
        [LoadColumn(0)]
        public string UserAbility { get; set; }
        [LoadColumn(1)]
        public string QuestionDifficulty { get; set; }

        [LoadColumn(2), ColumnName("Label")]
        public bool Answer;
    }

    public class QuestionPrediction
    {
        //will be a 1 for would know, or 0 for wouldn't know answer, for the new line input (unseen input)
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        //will give a probability value for the new line
        //[ColumnName("Probability")]
        public float Probability { get; set; }

        //another metric - both thus and the above are used for evaluating the model afterwards
        //[ColumnName("Score")]
        public float Score { get; set; }


    }
}
