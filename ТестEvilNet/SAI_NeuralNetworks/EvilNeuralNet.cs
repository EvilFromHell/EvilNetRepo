using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
namespace SAI_NeuralNetworks
{
    [Serializable]
    class EvilNeuralNet
    {
        public EvilNeuralNet(IList<int> LayersNeuronNumbers) //конструктор сети. На вход принимается список типа int. Каждое число в списке-это число нейронов в соответствующем слое
        {
            if (LayersNeuronNumbers == null || LayersNeuronNumbers.Count < 2 || LayersNeuronNumbers.Any(x => x <= 0)) throw new ArgumentException("Количество слоев не может быть меньше двух. В данное число входит входной и выходной слой. Так же не может быть слоев, состоящих из отрицательного количества нейронов.");
            List<Neuron> CurrentLayer = new List<Neuron>();
            Random rnd = new Random();
            for (int i = 0; i < LayersNeuronNumbers.Count; i++) //создаем слои. Веса задаем рандомно.
            {
                for (int j = 0; j < LayersNeuronNumbers[i]; j++)
                {
                    Neuron ne = new Neuron();
                    if (Layers.Count > 0)
                    {
                        foreach (Neuron upperne in Layers.Last())
                        {
                            ne.UpConnection.Add(upperne, rnd.NextDouble() - 0.5);
                        }
                    }
                    CurrentLayer.Add(ne);
                }
                Layers.Add(CurrentLayer);
                CurrentLayer = new List<Neuron>();
            }
        }

        public List<List<Neuron>> Layers { get; set; } = new List<List<Neuron>>(); //список слоев
        public List<Neuron> FirstLayer { get { return Layers.FirstOrDefault(); }} //входной слой
        public List<Neuron> LastLayer { get { return Layers.LastOrDefault(); }} //выходной слой


        public List<double> CalculateResult(IList<double> InputData) //метод, высчитывающий результат
        {
            if (InputData.Count != FirstLayer.Count) throw new ArgumentException("Количество входных значений должно быть равно количеству входных нейронов у сети");
            for (int i = 0; i < InputData.Count; i++) //вбиваем данные во входной слой
            { 
                FirstLayer[i].Value = InputData[i];
            }
            for (int i = 1; i < Layers.Count; i++)
            {
                for (int j = 0; j < Layers[i].Count; j++)
                {
                    Layers[i][j].ActivationFunc(); //вызываем для нейрона активационную функцию, тем самым меняя его значение
                }
            }
            return LastLayer.Select(x => x.Value).ToList();
        }

        public double Learn(IList<double> InputData, IList<double> CorrectOutputData, double speedCoef) //обучающий метод. Обратное распространение ошибки. Вовзращает ошибку.
        {
            CalculateResult(InputData);
            var delta = new double[LastLayer.Count];
            var newdelta = new double[LastLayer.Count];
            for (int i = 0; i < LastLayer.Count; i++)
            {
                var O = LastLayer[i].Value;
                delta[i] = (CorrectOutputData[i] - O) * O * (1 - O);
            }
            for (int k = Layers.Count - 1; k > 0; k--)
            {
                for (int i = 0; i < Layers[k].Count; i++)
                {
                    for (int j = 0; j < Layers[k - 1].Count; j++)
                    {
                        Layers[k][i].UpConnection[Layers[k - 1][j]] += speedCoef * delta[i] * Layers[k-1][j].Value;  //высчитываем веса слоя. Перемножаем коэффициент скорости обучения, дельту текущего нейрона и значение текущего нейрона. Все это добавляем к весу соттветствующей связи со слоем выше.
                    }
                }
                if (k > 1)
                {
                    newdelta = new double[Layers[k - 1].Count];
                    for (int i = 0; i < Layers[k - 1].Count; i++)
                    {
                        double s = 0;
                        for (int j = 0; j < Layers[k].Count; j++)
                        {
                            s += Layers[k][j].UpConnection[Layers[k - 1][i]] * delta[j];
                        }
                        newdelta[i] = Layers[k - 1][i].Value * (1 - Layers[k - 1][i].Value) * s;
                    }
                    delta = newdelta;
                }
            }
            return CalcError(CorrectOutputData);
        }


        public double CalcError(IList<double> Y) //вычисление ошибки, вычисляется она как сумма квадратов разницы между выходными сигналами сети и их требуемыхми значениями. 
        {
            double kErr = 0;
            for (int i = 0; i < Y.Count; i++)
            {
                kErr += Math.Pow(Y[i] - LastLayer[i].Value, 2);
            }
            return 0.5 * kErr;
        }

        [Serializable]
        public class Neuron
        {
            public Neuron()
            { }
            public double Value { get; set; } //значение нейрона
            public Dictionary<Neuron, double> UpConnection { get; set; } = new Dictionary<Neuron, double>();//связи со слоем выше и их веса
            public void ActivationFunc()//активационная функция
            {
                double s = 0;
                foreach (KeyValuePair<Neuron, double> p in this.UpConnection)
                {
                    s += p.Key.Value * p.Value;
                }
                s = 1.0 / (1 + Math.Exp(-s));
                this.Value = 0.998 * s + 0.001; 
            }
        }


        public static void Serialize(EvilNeuralNet net, string filename) //сериализация сети в файл
        {
            BinaryFormatter formatter = new BinaryFormatter();
            using (FileStream fs = new FileStream(filename, FileMode.Create))
                formatter.Serialize(fs, net);
        }

        public static EvilNeuralNet Deserialize(string filename)
        {
            BinaryFormatter formatter = new BinaryFormatter();
            using (FileStream fs = new FileStream(filename, FileMode.Open)) //десериализация из файла
                return formatter.Deserialize(fs) as EvilNeuralNet;
        }
    }
}
