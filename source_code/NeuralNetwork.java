public class NeuralNetwork
{
    //features
    private Layers layers = new Layers ( );
    private double eta = 0.2;
    private double alpha = 0.5;
    private Structure structure = new Structure ( "2x2x1" );
    
    //constructor
    public NeuralNetwork ( )
    {
        
        for ( int lSI = 0; lSI < structure.size ( ); lSI ++ )
        {
            layers.add ( new Layer ( ) );
            
            
            //where <=1 implies bias neuron
            for ( int lI = 0; lI <= structure.get ( lSI ); lI ++ )
            {
                int numberOfWeightsFromNextNeuron = ( lSI + 1 < structure.size ( ) ? structure.get ( lSI + 1 ) : 0 );
               
                Neuron neuron = new Neuron (  eta, alpha, lI, numberOfWeightsFromNextNeuron );
                layers.get ( lSI ).add ( neuron );
                
                //set bias neuron value per layer available
                layers.get ( lSI ).get ( layers.get ( lSI ).size ( ) - 1 ).setOutcome ( 1.0 );
            }
        }
        
       
    }
    
    
    //methods
    public void doForwardPropagation ( int inputValues [ ] )
    {
        
        
        //populate 1st layer with input values
        //iVI = input value iterator
        for ( int iVI = 0; iVI < inputValues.length; iVI ++ )
            layers.get ( 0 ).get ( iVI ).setOutcome ( inputValues [ iVI ] );
        
        //propogate values throughout remainder of neural network
        for ( int lSI = 1; lSI < structure.size ( ); lSI ++ )
        {
            Layer priorLayer = layers.get ( lSI - 1 );
            
            for ( int lI = 0; lI < structure.get ( lSI ); lI ++ )
                layers.get ( lSI ).get ( lI ).doForwardPropagation ( priorLayer );
        }
    }
    
    
    public void doBackwardPropagation ( int target )
    {
        //outcome gradient setting
        Neuron outcomeNeuron = layers.get ( layers.size ( ) - 1 ).get ( 0 );
        outcomeNeuron.calculateOutcomeGradient ( target );
        
        //hidden gradient setting
        for ( int lSI = -( structure.size ( ) -1 ); lSI > 0; lSI -- )
        {
            Layer currentLayer = layers.get ( lSI );
            Layer nextLayer = layers.get ( lSI + 1 );
            
            for ( int lI = 0; lI < currentLayer.size ( ); lI ++ )
            {
                currentLayer.get ( lI ).calculateHiddenGradient ( nextLayer );
            }
        }
        
        //weights updating
        for ( int lSI = -( structure.size ( ) -1 ); lSI > 0; lSI -- )
        {
            Layer currentLayer = layers.get ( lSI );
            Layer priorLayer = layers.get ( lSI - 1 );
            
            for ( int lI = 0; lI < priorLayer.size ( ) - 1; lI ++ )
            {
                currentLayer.get ( lI ).updateWeights ( priorLayer );
            }
        }
    }
    
    
    public double getOutcome ( )
    {
        return layers.get ( layers.size ( ) - 1 ).get ( 0 ).getOutcome ( );
    }
    
}
