using System.Collections;

namespace DeepSharp
{
    public class Dataset<T> : IEnumerable<LabeledData<T>> where T : Tensor
    {
        List<LabeledData<T>> _data;

        public Dataset()
        {
            _data = new List<LabeledData<T>>();
        }

        public Dataset(List<LabeledData<T>> data)
        {
            this._data = data;
        }

        public int Count => _data.Count;

        public void Add(LabeledData<T> item)
        {
            _data.Add(item);
        }

        public IEnumerator<LabeledData<T>> GetEnumerator()
        {
            return new DatasetEnumerator(_data);
        }
        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public void Dispose()
        {
            // No resources to dispose
        }

        private class DatasetEnumerator : IEnumerator<LabeledData<T>>
        {
            private readonly List<LabeledData<T>> _data;
            private int _index = -1;
            public DatasetEnumerator(List<LabeledData<T>> data)
            {
                _data = data;
            }
            public LabeledData<T> Current => _data[_index];
            object IEnumerator.Current => Current;
            public bool MoveNext()
            {
                _index++;
                return (_index < _data.Count);
            }
            public void Reset()
            {
                _index = -1;
            }
            public void Dispose()
            {
                // No resources to dispose
            }
        }
    }

    public class LabeledData<T> where T : Tensor
    {
        public T Data;
        public int Label;
        public LabeledData(T data, int label)
        {
            Data = data;
            Label = label;
        }
    }

    public class BatchedData<T> where T : BatchTensor
    {
        // 先頭行がバッチなTensor
        public T BatchTensor;
        public int[] Labels;

        public BatchedData(Tensor[] sampleTensors, int[] labels)
        {
            if (sampleTensors.Length != labels.Length)
                throw new ArgumentException("Data and Labels must have the same length.");

            var baseBatch = new BatchTensor(sampleTensors, name: sampleTensors[0].Name);
            if (!(baseBatch is T typedBatch))
                throw new ArgumentException($"sampleTensors cannot be converted to {typeof(T).Name}");

            this.BatchTensor = typedBatch;
            this.Labels = labels;
        }

        public BatchedData(BatchTensor batchTensor, int[] labels)
        {
            if (batchTensor.BatchSize != labels.Length)
                throw new ArgumentException("Batch size and Labels length must match.");

            if (!(batchTensor is T typedBatch))
                throw new ArgumentException($"Provided batchTensor is not of type {typeof(T).Name}");

            this.BatchTensor = typedBatch;
            this.Labels = labels;
        }
    }

    public enum DatasetType 
    { 
        Train,
        Validation,
        Test 
    }
}
