import java.util.ArrayList;

public class Structure extends ArrayList <Integer>
{
    //features
    private String structure = "";
    
    //constructor
    public Structure ( String structure )
    {
        this.structure = structure;
        
        establishStructure ( this.structure );
    }
    
    public void establishStructure ( String structure )
    {
        String [ ] structureData = structure.split ( "x" );
        
        for ( int sDI = 0; sDI < structureData.length; sDI ++ )
            add ( Integer.parseInt ( structureData [ sDI ] ) );
    }
}
