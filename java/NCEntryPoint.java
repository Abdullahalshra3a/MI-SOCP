import org.networkcalculus.dnc.AnalysisConfig;
import org.networkcalculus.dnc.curves.ArrivalCurve;
import org.networkcalculus.dnc.curves.Curve;
import org.networkcalculus.dnc.curves.ServiceCurve;
import org.networkcalculus.dnc.network.server_graph.Flow;
import org.networkcalculus.dnc.network.server_graph.Server;
import org.networkcalculus.dnc.network.server_graph.ServerGraph;
import org.networkcalculus.dnc.tandem.TandemAnalysis;
import org.networkcalculus.dnc.tandem.analyses.TandemMatchingAnalysis;
import py4j.GatewayServer;

import java.io.FileWriter;
import java.util.Map;
import java.util.List;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Order of calls to make for a fully functioning NC calculation:
 * <p><ol>
 * <li> {@link #addEdge(String, String, double, double)} [multiple times] - Add your edges of the network to the Servergraph (Edge = Output-link of one hop)
 * <li> {@link #addSGService(String, String, int, int, double, List, int)} [multiple times] - Add all the SGS that will use the network. Flows are aggregated in here.
 * <li> {@link #createNCNetwork()} [once] - Create the final Servergraph with every connection in it.
 * <li> {@link #calculateNCDelays()} - Call this function for calculating the final delays per flow.
 * </ol>
 */

public class NCEntryPoint {
    private final List<Edge> edgeList = new ArrayList<>();
    private final ExperimentConfig experimentConfig = new ExperimentConfig();
    private List<SGService> sgServices = new ArrayList<>();
    private ServerGraph serverGraph;

    private static final String LATEX_FILE_NAME = "max_service_delays.tex";


    public NCEntryPoint() {
    }

    /**
     * Retrieve all connected neighbors of a specific edge
     *
     * @param currEdge edge, for which the neighbors shall be found
     * @param edgeList list of edges to search in.
     * @return list of found neighbors.
     */
    private static List<Edge> getAllConnectingEdges(Edge currEdge, Collection<Edge> edgeList) {
        List<Edge> targetEdgeList;
        HashSet<String> currEdgeNodes = new HashSet<>(currEdge.getNodes());
        /*
         * Check if the edges are connected. They are connected if the last node in an edge is the first node in the second node
         * e.g. R10/R20 & R20/R30
         * The first two comparisons: Check for connecting node
         * The third line: Check that the two compared edges do not concern the same node pairs
         *                 (aka are the same edge but maybe different direction --> R10/R20 & R20/R10)
         */
        targetEdgeList = edgeList.stream()
                .filter(edge -> (edge.getNodes().get(0).equals(currEdge.getNodes().get(1)) ||
                        edge.getNodes().get(1).equals(currEdge.getNodes().get(0)))
                        && !currEdgeNodes.containsAll(edge.getNodes()))
                .collect(Collectors.toList());
        return targetEdgeList;
    }

    /**
     * Create noPrios service curves for one edge, according to the scheduling
     * and other configuration parameters defined in expConfig
     *
     * @param edge      Edge for which the service curves shall be created. Needed for bitrate
     * @param expConfig Experiment config to use, including priorities and scheduling parameters
     * @param noPrios   number of priority service curves shall be created
     * @return List of Service curves in increasing priority order
     */
    private static List<ServiceCurve> createServiceCurves(Edge edge, ExperimentConfig expConfig, int noPrios) {
        // Define base service curve for this server
        double rate;
        double latency = 0;
        List<ServiceCurve> serviceCurves = new ArrayList<>();

        switch (expConfig.schedulingPolicy) {
            case None -> serviceCurves = createSimpleServiceCurves(edge, expConfig, noPrios);
            case SP -> {
                // The strict priority service curve is the link-service curve - the cross-traffic arrival
                // ==> The cross-traffic subtraction is done at other ends
                rate = edge.getBitrate();
                if (expConfig.usePacketizer) {
                    //  The packetized SP is b_SP = b - a - l_max [- l_max]
                    //  One (- l_max) because we have to wait for the lower priority packet which gets served
                    //  --> not for the lowest priority
                    //  And one (- l_max) to account for the packetizer (transmission delay)
                    latency = expConfig.maxPacketSize / edge.getBitrate();
                }
                // latency is 0 for non-packetized SP
                for (int i = 0; i < noPrios - 1; i++) {
                    // For all priorities unequal the lowest priority, b = b - a - l_max*2
                    serviceCurves.add(Curve.getFactory().createRateLatency(rate, 2 * latency));
                }
                // For the lowest priority it would only be one time l_max (transmission delay)
                serviceCurves.add(Curve.getFactory().createRateLatency(rate, latency));
            }
            case WFQ -> {
                if (expConfig.usePacketizer) {
                    latency = (2 * expConfig.maxPacketSize) / edge.getBitrate();
                }
                double sumWeights = Arrays.stream(expConfig.flowWeights).sum();
                for (int i = 0; i < noPrios; i++) {
                    rate = (expConfig.flowWeights[i] / sumWeights) * edge.getBitrate();
                    serviceCurves.add(Curve.getFactory().createRateLatency(rate, latency));
                }
            }
            case DRR -> {
                int l_max = expConfig.maxPacketSize;
                int L = l_max * noPrios;
                double F = Arrays.stream(expConfig.flowQuantils).sum();
                double C = edge.getBitrate();
                for (int i = 0; i < noPrios; i++) {
                    int Q_i = expConfig.flowQuantils[i];
                    latency = ((Q_i * (L - l_max)) + ((F - Q_i) * (Q_i + l_max)) + (Q_i * l_max)) / (Q_i * C);
                    rate = (Q_i / F) * C;
                    serviceCurves.add(Curve.getFactory().createRateLatency(rate, latency));
                }
            }
            case WRR -> {
                int l_min = expConfig.minPacketSize;
                int l_max = expConfig.maxPacketSize;
                for (int i = 0; i < noPrios; i++) {
                    int w_i = expConfig.flowWeights[i];
                    double q_i = w_i * l_min;
                    double Q_i = (Arrays.stream(expConfig.flowWeights).sum() - w_i) * l_max;
                    latency = (Q_i + l_max) / edge.getBitrate();
                    rate = (q_i / (q_i + Q_i)) * edge.getBitrate();
                    serviceCurves.add(Curve.getFactory().createRateLatency(rate, latency));
                }
            }
            case SCFQ -> {
            // Self-Clocked Fair Queueing
            double virtualTime = 0;  // The virtual time to track system progress
            double sumWeights = Arrays.stream(expConfig.flowWeights).sum();
            double bitrate = edge.getBitrate();

            // Iterate over each flow/priority
            for (int i = 0; i < noPrios; i++) {
                // Calculate the rate proportional to the weight
                rate = (expConfig.flowWeights[i] / sumWeights) * bitrate;

                // In SCFQ, we need to adjust the latency based on the virtual time
                // Assuming packets arrive at different times, we simulate the time the packet would finish being served
                latency = (1 / rate) + virtualTime; // Adjust based on virtual time and service rate

                // Update virtual time for the next priority
                virtualTime += (expConfig.flowWeights[i] / bitrate);

                // Add the service curve for this priority
                serviceCurves.add(Curve.getFactory().createRateLatency(rate, latency));
            }
        }
    }
        return serviceCurves;
    }

    /**
     * Create noPrios simple service curves for one edge, not taking any scheduling algorithm into account
     *
     * @param edge      Edge for which the service curves shall be created. Needed for bitrate
     * @param expConfig Experiment config to use, including priorities and scheduling parameters
     * @param noPrios   number of service curves which shall be created
     * @return List of Service curves in increasing priority order
     */
    private static List<ServiceCurve> createSimpleServiceCurves(Edge edge, ExperimentConfig expConfig, int noPrios) {
        double latency = 0;
        List<ServiceCurve> serviceCurves = new ArrayList<>();
        // model the link simply as a combination of the packet burst + link rate
        if (expConfig.usePacketizer) {
            latency = expConfig.maxPacketSize / edge.getBitrate();
        }

        if (expConfig.useGivenLinkDelay) {
            //TODO: Talk with Kai-Steffen if we should delete that
            latency += edge.getLatency();
        }
        double rate = edge.getBitrate();
        for (int i = 0; i < noPrios; i++) {
            serviceCurves.add(Curve.getFactory().createRateLatency(rate, latency));
        }
        return serviceCurves;
    }

    /**
     * Function used to conduct tests for the service curves of a simple one hop, two server network
     * The following results should be acquired (same result for SFA and PMOO):
     * SP:
     * SFA: 5100ms
     * TFA: 6693,75ms
     * WFQ (all weights = 1, 3 priorities, l_max = 255)
     * SFA: 7650ms
     * TFA: 9881,25ms
     * WRR (all weights = 1, 3 priorities, l_min = l_max = 255)
     * SFA: 8925ms
     * DRR (all Q_i = 1*l_max = 255, 3 priorities)
     * SFA: 14025ms
     */
    @SuppressWarnings("unused")
    private static void testSimpleNetwork() {
        NCEntryPoint entryPoint = new NCEntryPoint();
        entryPoint.addEdge("F1", "H1", 200, 10);
        entryPoint.addEdge("H1", "S1", 200, 10);

        List<List<String>> path_list = new ArrayList<>();
        path_list.add(Arrays.asList("F1", "H1", "S1"));

        entryPoint.addSGService("SGTest_high", "S1", 255, 50, 1000, path_list, 0);
        entryPoint.addSGService("SGTest_med", "S1", 255, 50, 1000, path_list, 1);
        entryPoint.addSGService("SGTest_low", "S1", 255, 50, 1000, path_list, 2);
        entryPoint.createNCNetwork();
        entryPoint.calculateNCDelays();
    }

    public static void main(String[] args) {
        GatewayServer gatewayServer = new GatewayServer(new NCEntryPoint());
        gatewayServer.start();
        System.out.println("Gateway Server Started");
    }

    //TODO: Check for better Exception handling (here "sg.addTurn()" throws exception if servers not present etc.)

    /**
     * Retrieves an edge out of a list, defined by its node pair.
     *
     * @param listEdge edge list to search in
     * @param nodes    node collection to use for comparison with every edge.
     * @return matching edge, null if none is found.
     */
    public static Edge findEdgebyNodes(Collection<Edge> listEdge, Collection<String> nodes) {
        return listEdge.stream().filter(edge -> edge.getNodes().equals(nodes)).findFirst().orElse(null);
    }

    /**
     * This function is called via the py4j Gateway from the python source code.
     * Every Edge is added to the network list one by one. (The node order is not yet defined)
     * Bitrate + latency will be later used for modeling the link as a service curve
     *
     * @param node1   first node
     * @param node2   second node
     * @param bitrate link bitrate [Byte/s]
     * @param latency link delay [s]
     */
    @SuppressWarnings("unused")
    public void addEdge(String node1, String node2, double bitrate, double latency) {
        Edge newEdge = new Edge(node1, node2, bitrate, latency);
        edgeList.add(newEdge);
    }

    /**
     * Add SGS to the list. To be called via Python.
     *
     * @param SGSName     SGService name
     * @param servername  server name where the service is running on
     * @param bucket_size bucket size for token bucket arrival curve modeling.
     * @param bitrate     bitrate for token bucket arrival curve modeling
     * @param deadline    deadline of the service (in ms)
     * @param multipath   all paths which are used for the flows
     * @param priority    Priority of the SGS, the highest priority is 0
     */
    @SuppressWarnings("unused")
    public void addSGService(String SGSName, String servername, int bucket_size, int bitrate, double deadline, List<List<String>> multipath, int priority) {
        // 0 is always the highest priority
        if (priority < 0) {
            priority = 0;
        }
        if (priority >= FlowPriority.values().length) {
            priority = FlowPriority.values().length - 1;    // lowest possible priority
        }
        FlowPriority priority_enum = FlowPriority.values()[priority];
        SGService service = new SGService(SGSName, servername, bucket_size, bitrate, deadline, multipath, priority_enum);
        sgServices.add(service);
    }

    /**
     * Reset all stored values (e.g. empty edgelist)
     */
    @SuppressWarnings("unused")
    public void resetAll() {
        edgeList.clear();
        sgServices.clear();
    }

    /**
     * This function creates the final ServerGraph aka combines all the network elements in one network.
     * Has to be called last, AFTER calling addEdge and addSGService for adding the network elements.
     */
    @SuppressWarnings("unused")
    public void createNCNetwork() {
        // Create ServerGraph - aka network
        ServerGraph sg = new ServerGraph();

        // Add every edge as a server to the network
        for (Edge edge : edgeList) {
            List<String> edgeNodes = edge.getNodes();
            List<ServiceCurve> service_curves;
            // When a field device is involved, only a simple service curve shall be created instead of a scheduling one.
            if (edgeNodes.get(0).contains("F") || edgeNodes.get(1).contains("F")) {
                service_curves = createSimpleServiceCurves(edge, experimentConfig, FlowPriority.values().length);
            } else {
                // Create the service curve according to the current configuration settings
                service_curves = createServiceCurves(edge, experimentConfig, FlowPriority.values().length);
            }
            // Add server (edge) with service curve to network
            // (Important: Every "Edge"/"Server" in this Java code is unidirectional - not bidirectional!)
            // --> For two-way /bidirectional but independent communication (e.g. switched Ethernet) use the "addEdge"
            // function twice with a switched order of nodes.
            for (int i = 0; i < FlowPriority.values().length; i++) {
                FlowPriority prio = FlowPriority.values()[i];
                String servername = String.join(",", edge.getNodes()) + prio;
                Server serv = sg.addServer(servername, service_curves.get(i), experimentConfig.multiplexing);
                // Add server to edge for future references
                // IMPORTANT: The servers have to be added in ascending priority order (HIGH before MEDIUM or LOW)!
                edge.setServer(prio, serv);
            }
        }
        // Add the turns (connections) between the edges to the network
        addTurnsToSG(sg);

        // Add all flows to the network
        FlowPriority fixedPrio = null;
        if (experimentConfig.schedulingPolicy == ExperimentConfig.SchedulingPolicy.None){
            fixedPrio = FlowPriority.values()[0];   // just take the first prio to add all flows to.
        }
        addFlowsToSG(sg, sgServices, -1, fixedPrio);
        this.serverGraph = sg;
        System.out.printf("%d Flows %n", sg.getFlows().size());
    }

    /**
     * Helper function for adding the turn connections between the edges into a given SererGraph
     *
     * @param sg SerGraph to add all the connections to.
     */
    private void addTurnsToSG(ServerGraph sg) {
        for (Edge currEdge : edgeList) {
            List<Edge> targetEdgeList = getAllConnectingEdges(currEdge, edgeList);
            for (Edge targetEdge : targetEdgeList) {
                // We can just freely add one turn twice, duplicates get omitted by DiscoDNC
                try {
                    // Connect the NC servers according to their priorities --> No priority hoping possible!
                    for (FlowPriority prio : FlowPriority.values()) {
                        sg.addTurn(currEdge.getServer(prio), targetEdge.getServer(prio));
                    }
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }

    @SuppressWarnings("unused")
    public boolean experimentOneCombination() {
        boolean delayTorn = true;
        // Start the timer to measure execution time
        long startTime = System.nanoTime();  // Use System.currentTimeMillis() if you prefer milliseconds

        // The experimentLog is one List<String> with each entry being a String "," separated containing each experiment
        List<List<String>> experimentLog = new ArrayList<>();

        // Assign one multiplexing technique (FIFO or ARBITRARY)
        var multiplexing = AnalysisConfig.Multiplexing.FIFO; // Chosen value, for example FIFO
        experimentConfig.multiplexing = multiplexing;

        // Assign one network analysis method (TFA, SFA, PMOO, TMA)
        TandemAnalysis.Analyses anaType = TandemAnalysis.Analyses.TFA; // Chosen value, for example TFA
        if (multiplexing == AnalysisConfig.Multiplexing.FIFO &&
                ((anaType == TandemAnalysis.Analyses.PMOO) || (anaType == TandemAnalysis.Analyses.TMA))) {
            // PMOO & TMA don't support FIFO multiplexing, skip them
            return delayTorn;
        }
        experimentConfig.ncAnalysisType = anaType;

        // Assign one Arrival bounding method (AGGR_PBOO_PER_SERVER, AGGR_PBOO_CONCATENATION, AGGR_PMOO, AGGR_TM, SEGR_PBOO, SEGR_PMOO, SEGR_TM, SINKTREE_AFFINE_MINPLUS, SINKTREE_AFFINE_DIRECT, SINKTREE_AFFINE_HOMO)
        var arrBoundType = AnalysisConfig.ArrivalBoundMethod.AGGR_PBOO_PER_SERVER; // Chosen value
        if (arrBoundType == AnalysisConfig.ArrivalBoundMethod.SEGR_TM) {
            // Skip SEGR_TM as it lets the program crash
            return delayTorn;
        }
        if (multiplexing == AnalysisConfig.Multiplexing.FIFO &&
                (arrBoundType == AnalysisConfig.ArrivalBoundMethod.AGGR_TM ||
                        arrBoundType == AnalysisConfig.ArrivalBoundMethod.SEGR_PMOO ||
                        arrBoundType == AnalysisConfig.ArrivalBoundMethod.AGGR_PMOO)) {
            return delayTorn;
        }
        experimentConfig.arrivalBoundMethod = arrBoundType;

        // Assign one scheduling policy (None, SP, WFQ, DRR, WRR, SCFQ)
        var schedPol = ExperimentConfig.SchedulingPolicy.SCFQ; // Chosen value, for example WFQ
        experimentConfig.schedulingPolicy = schedPol;

        // Reset the old server graph, modify the service curves and flow paths according to the used scheduler
        resetServerGraph();

        // Create the new NC network
        createNCNetwork();

        // Conduct the experiment with the newly defined configurations
        List<String> buffer = new ArrayList<>();
        delayTorn = calculateNCDelays(buffer);
        experimentLog.add(buffer);

        experimentConfig.insertConfigNamesInFront(experimentLog);
        exportResultToCSV(experimentLog, "experiments", "experiment");

        // End the timer and calculate the elapsed time
        long endTime = System.nanoTime();
        long duration = (endTime - startTime);  // Duration in nanoseconds
        System.out.println("Experiment finished in " + duration / 1_000_000 + " milliseconds.");
        System.out.println(delayTorn);
        return delayTorn;
    }

    /**
     * This function tries every network analysis method, combined with every arrival bounding technique
     * The result will be exported to the newly created folder "experiments" as a CSV file.
     */
    @SuppressWarnings("unused")
    public void experimentAllCombinations() {
        // The experimentLog is one List<String> with each entry being a String "," separated containing each experiment
        List<List<String>> experimentLog = new ArrayList<>();
        // Iterate over every multiplexing technique (FIFO & ARBITRARY)
        for (var multiplexing : AnalysisConfig.Multiplexing.values()) {
            experimentConfig.multiplexing = multiplexing;

            // Iterate over every network analysis method// TFA, SFA, PMOO, TMA
            for (TandemAnalysis.Analyses anaType : TandemAnalysis.Analyses.values()) {
                if (multiplexing == AnalysisConfig.Multiplexing.FIFO &&
                        ((anaType == TandemAnalysis.Analyses.PMOO) || (anaType == TandemAnalysis.Analyses.TMA))) {
                    // PMOO & TMA don't support FIFO multiplexing, skip them
                    continue;
                }
                experimentConfig.ncAnalysisType = anaType;

                // Iterate over every Arrival bounding method => AGGR_PBOO_PER_SERVER, AGGR_PBOO_CONCATENATION, AGGR_PMOO, AGGR_TM, SEGR_PBOO, SEGR_PMOO, SEGR_TM, SINKTREE_AFFINE_MINPLUS, SINKTREE_AFFINE_DIRECT, SINKTREE_AFFINE_HOMO
                for (var arrBoundType : AnalysisConfig.ArrivalBoundMethod.values()) {
                    // SEGR_TM lets the program crash
                    if (arrBoundType == AnalysisConfig.ArrivalBoundMethod.SEGR_TM) {
                        continue;
                    }
                    // These arrival boundings are not available with FIFO
                    if(multiplexing == AnalysisConfig.Multiplexing.FIFO &&
                            (arrBoundType == AnalysisConfig.ArrivalBoundMethod.AGGR_TM ||
                                    arrBoundType == AnalysisConfig.ArrivalBoundMethod.SEGR_PMOO ||
                                    arrBoundType == AnalysisConfig.ArrivalBoundMethod.AGGR_PMOO)) {
                        continue;
                    }
                    experimentConfig.arrivalBoundMethod = arrBoundType;

                    // Iterate over every scheduling policy
                    // Assign one scheduling policy (None, SP, WFQ, DRR, WRR, SCFQ)
                    var schedPol = ExperimentConfig.SchedulingPolicy.SCFQ; // Chosen value, for example WFQ
                    experimentConfig.schedulingPolicy = schedPol;

                    // Reset the old servergraph, we need to modify the service curves and flow paths
                    // according to the used scheduler
                    resetServerGraph();

                    // Create the new NC network
                    createNCNetwork();

                    // conduct the experiment with the newly defined configurations
                    List<String> buffer = new ArrayList<>();
                    calculateNCDelays(buffer);
                    experimentLog.add(buffer);
                    }

            }
        }
        experimentConfig.insertConfigNamesInFront(experimentLog);
        exportResultToCSV(experimentLog, "experiments", "experiment");
        closeLatexFile();


    }


    private static void exportResultToCSV(List<List<String>> experimentLog, String folderName, String prefix) {
        // Export experimentLog to a file
        try {
            String fileSuffix = new SimpleDateFormat("yyyyMMddHHmmss").format(new Date());
            // Create the experiments' subfolder, if not present
            File directory = new File(folderName);
            if (!directory.exists()) {
                boolean success = directory.mkdir();
                if (!success){
                    System.err.println("Error when creating folder for exporting results. Aborting export.");
                    return;
                }
            }
            String filename = folderName + "/" + prefix + "Log_" + fileSuffix + ".csv";
            FileWriter writer = new FileWriter(filename);
            for (List<String> row : experimentLog){
                String row_str = String.join(";", row);
                writer.write(row_str + System.lineSeparator());
            }
            writer.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Function called by Python part.
     * For details see {@link #calculateNCDelays(List)}
     *
     * @return boolean if one of the delay constraints is torn
     */
    @SuppressWarnings("UnusedReturnValue")
    public boolean calculateNCDelays() {
        List<List<String>> experimentLog = new ArrayList<>();
        List<String> buffer = new ArrayList<>();

        boolean delayTorn =  calculateNCDelays(buffer);
        experimentLog.add(buffer);
        experimentConfig.insertConfigNamesInFront(experimentLog);
        exportResultToCSV(experimentLog, "calcs", "bounding");
        return delayTorn;
    }

    /**
     * Function to calculate the performance bounds for every flow and output the result onto the Console and into a
     * given experimentLog
     *
     * @param experimentLog List to add the output results into. Intended for CSV usage.
     * @return boolean if one of the delay constraints is torn
     */
    public boolean calculateNCDelays(List<String> experimentLog) {
        // The AnalysisConfig can be used to modify different analysis parameters, e.g. the used arrival bounding method
        // or to enforce Multiplexing strategies on the servers.
        AnalysisConfig configuration = new AnalysisConfig();
        configuration.setArrivalBoundMethod(experimentConfig.arrivalBoundMethod);
        // Current work-around for the experiment run
        if (experimentConfig.multiplexing == AnalysisConfig.Multiplexing.ARBITRARY) {
            configuration.enforceMultiplexing(AnalysisConfig.MultiplexingEnforcement.GLOBAL_ARBITRARY);
        } else {
            configuration.enforceMultiplexing(AnalysisConfig.MultiplexingEnforcement.SERVER_LOCAL);
        }
        experimentConfig.outputConfig();
        experimentConfig.writeConfiginBuffer(experimentLog);
        expLog_addSGSPrios(experimentLog, sgServices);
        try {
            System.out.printf("------ Starting NC Analysis using " + experimentConfig.ncAnalysisType + " ------%n");
            if (experimentConfig.schedulingPolicy == ExperimentConfig.SchedulingPolicy.SP) {
                return calculate_SP_Delays(configuration, experimentConfig, experimentLog);
            }
            Map<String, List<Double>> perf_results = new HashMap<>();
            boolean delayTorn = conductNC_Analysis(perf_results, configuration, sgServices, experimentConfig);
            convertPerfResultsExpLog(experimentLog, perf_results);

            return  delayTorn;
        } catch (StackOverflowError e) {
            System.err.println("Stackoverflow error detected! Possible reason: Cyclic dependency in network.");
            return true;
        }
    }

    /**
     * Add the priorities of the different SGS to the experiment log.
     * @param experimentLog experiment log in which the priorities shall be added
     * @param sgServices Considered SGS from where the priorities shall be gathered
     */
    private static void expLog_addSGSPrios(List<String> experimentLog, List<SGService> sgServices) {
        StringBuilder result = new StringBuilder();
        String prefix = "";
        for (var sgs : sgServices){
            result.append(prefix);
            result.append(sgs.getName());
            result.append(":");
            result.append(sgs.getPriority());
            prefix = " - ";
        }

        experimentLog.add(result.toString());
    }

    /**
     * Internal function for conversion between Map and CSV export.
     * @param experimentLog CSV intended column List
     * @param perf_results Evaluation results in map format
     */
    private static void convertPerfResultsExpLog(List<String> experimentLog, Map<String, List<Double>> perf_results) {
        List<String> sgs_keys = new ArrayList<>(perf_results.keySet());
        Collections.sort(sgs_keys);
        for (String sgs : sgs_keys){
            experimentLog.add(sgs);
            for (Double flow_delay : perf_results.get(sgs)) {
                experimentLog.add(String.format("%.3f", flow_delay));
            }
        }
    }

    /**
     * Function to perform an NC analysis for strict priority scheduling, following the step-by-step algorithm
     *
     * @param analysisConfig   DiscoDNC network analysis configuration
     * @param experimentConfig Overall experiment configuration, containing the parameters
     * @param experimentLog    List to add the output results into. Intended for CSV usage.
     * @return boolean if one of the delay constraints is torn
     */
    private boolean calculate_SP_Delays(AnalysisConfig analysisConfig, ExperimentConfig experimentConfig, List<String> experimentLog) {
        // First: Delete all present flows from the serverGraph (e.g. from previous analysis)
        removeAllFlows();
        // Procedure: add highest prio, calculate delay. Add next prio, calculate again
        boolean highest_prio_finished = false;
        boolean delayTorn = false;
        Map<String, List<Double>> perf_results = new HashMap<>();
        List<SGService> curr_SGSs = new ArrayList<>();
        for (FlowPriority prio : FlowPriority.values()) {
            // Select all SGSs which have the current priority
            List<SGService> currprioSGSs = this.sgServices.stream().filter(sgService -> sgService.getPriority() == prio).toList();

            // Add the new flows to the flows in the server graph.
            curr_SGSs.addAll(currprioSGSs);

            // Add all flows of this priority to the network
            this.addFlowsToSG(this.serverGraph, curr_SGSs, -1, prio);

            // Change the multiplexing technology - only for the highest priority we can use the multiplexing technology
            // specified in the experimentConfig, for other priorities we have to use Arbitrary
            if (!highest_prio_finished) {
                highest_prio_finished = true;
            } else {
                analysisConfig.enforceMultiplexing(AnalysisConfig.MultiplexingEnforcement.GLOBAL_ARBITRARY);
            }

            // Analyze performance of those flows
            boolean prioDelayTorn = conductNC_Analysis(perf_results, analysisConfig, currprioSGSs, experimentConfig);
            // Update delayTorn only to true
            delayTorn = delayTorn | prioDelayTorn;

            // Delete the flows from the server graph
            removeAllFlows();
        }
        convertPerfResultsExpLog(experimentLog, perf_results);
        return delayTorn;
    }

    /**
     * Helper function for conducting the DiscoDNC network analysis for every specified flow saved in sgServices
     *
     * @param results          Hashmap into which the results of the different SGSs will be stored in.
     * @param analysisConfig   DiscoDNC analysis configuration
     * @param sgServices       SGSs to be analyzed
     * @param experimentConfig Overall experimentConfiguration with the parameters
     * @return boolean if one of the delay constraints is torn
     */
    private boolean conductNC_Analysis(Map<String, List<Double>> results, AnalysisConfig analysisConfig, List<SGService> sgServices, ExperimentConfig experimentConfig) {
        boolean delayTorn = false;
        String configKey = experimentConfig.multiplexing + "-" + experimentConfig.ncAnalysisType + "-" + experimentConfig.arrivalBoundMethod;

        // Loop through all SGServices and analyze delays
        for (SGService sgs : sgServices) {
            double maxDelay = 0;
            System.out.printf("--- Analyzing SGS \"%s\" ---%n", sgs.getName());

            List<Double> flowDelays = new ArrayList<>();
            for (Flow foi : sgs.getFlows()) {
                System.out.printf("- Analyzing flow \"%s\" -%n", foi);
                try {
                    TandemAnalysis ncanalysis = switch (experimentConfig.ncAnalysisType) {
                        case TFA -> TandemAnalysis.performTfaEnd2End(this.serverGraph, analysisConfig, foi);
                        case SFA -> TandemAnalysis.performSfaEnd2End(this.serverGraph, analysisConfig, foi);
                        case PMOO -> TandemAnalysis.performPmooEnd2End(this.serverGraph, analysisConfig, foi);
                        case TMA -> new TandemMatchingAnalysis(this.serverGraph, analysisConfig);
                    };
                    if (experimentConfig.ncAnalysisType == TandemAnalysis.Analyses.TMA) {
                        ncanalysis.performAnalysis(foi);
                    }

                    double foi_delay = ncanalysis.getDelayBound().doubleValue();
                    double prop_delay = experimentConfig.propagationDelay * foi.getPath().numServers();
                    foi_delay += prop_delay;

                    System.out.printf("delay bound     : %.2fms %n", foi_delay * 1000);
                    System.out.printf("backlog bound   : %.2f %n", ncanalysis.getBacklogBound().doubleValue());

                    // Update max backlog bound for this configuration and service
                    updateMaxBacklogBound(configKey, sgs.getName(), ncanalysis.getBacklogBound().doubleValue());

                    flowDelays.add(foi_delay * 1000); // Convert s to ms
                    maxDelay = Math.max(foi_delay, maxDelay);
                } catch (Exception e) {
                    System.out.println(experimentConfig.ncAnalysisType + " analysis failed");
                    e.printStackTrace();
                    flowDelays.add(-1.0);
                }
            }
            results.put(sgs.getName(), flowDelays);

            System.out.printf("Max service delay for %s is %.2fms (deadline: %.2fms) %n", sgs.getName(), maxDelay * 1000, sgs.getDeadline() * 1000);
            if (sgs.getDeadline() < maxDelay) {
                System.err.printf("Service %s deadline not met (%.2fms/%.2fms) %n", sgs.getName(), maxDelay * 1000, sgs.getDeadline() * 1000);
                delayTorn = true;
            }
        }

        // After analyzing delays for the current experiment configuration, export the results to LaTeX
        exportToLatex(results, experimentConfig.multiplexing.toString(), experimentConfig.ncAnalysisType.toString(), experimentConfig.arrivalBoundMethod.toString());
        // After all configurations have been analyzed
        exportAllMaxBacklogBoundsToCSV("all_max_backlog_bounds.csv");
        return delayTorn;
    }


    private Map<String, StringBuilder> latexTables = new HashMap<>();
    private Map<String, Map<String, Double>> allMaxBacklogBounds = new HashMap<>();

    private void updateMaxBacklogBound(String configKey, String serviceName, double backlogBound) {
        allMaxBacklogBounds
                .computeIfAbsent(configKey, k -> new HashMap<>())
                .merge(serviceName, backlogBound, Math::max);
    }

    public void exportAllMaxBacklogBoundsToCSV(String fileName) {
        try (FileWriter writer = new FileWriter(fileName)) {
            // Write header
            writer.write("Configuration,AR,CVC,LM,SE,VPP\n");

            // Write data for each configuration
            for (Map.Entry<String, Map<String, Double>> entry : allMaxBacklogBounds.entrySet()) {
                String configKey = entry.getKey();
                Map<String, Double> bounds = entry.getValue();

                writer.write(configKey);
                for (String service : Arrays.asList("AR", "CVC", "LM", "SE", "VPP")) {
                    double maxBacklog = bounds.getOrDefault(service, 0.0);
                    writer.write(String.format(",%.2f", maxBacklog));
                }
                writer.write("\n");
            }
        } catch (IOException e) {
            System.err.println("Error writing to CSV file: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // Helper function to get the value or return "null" if key or value is missing
    public String getValueOrNull(Map<String, List<Double>> map, String key) {
        return (map.containsKey(key) && !map.get(key).isEmpty()) ? String.format("%.2f", map.get(key).get(0)) : "null";
    }



    private boolean isLatexFileInitialized = false;
    private String combinedFileName = "combined_experiment_results.tex";

    public void exportToLatex(Map<String, List<Double>> results, String multiplexing, String ncAnalysisType, String arrivalBoundMethod) {
        try (FileWriter writer = new FileWriter(combinedFileName, true)) {  // Open file in append mode

            // Write the LaTeX document header only once
            if (!isLatexFileInitialized) {
                writer.write("\\documentclass{article}\n");
                writer.write("\\usepackage{booktabs}\n");  // For better looking tables
                writer.write("\\begin{document}\n");
                isLatexFileInitialized = true;
            }

            // Write results for the current experiment configuration
            writer.write("\\section*{NC Analysis Results}\n");
            writer.write("Multiplexing: " + multiplexing + " \\\\\n");
            writer.write("Network Analysis Method: " + ncAnalysisType + " \\\\\n");
            writer.write("Arrival Bounding Method: " + arrivalBoundMethod + " \\\\\n");
            writer.write("\\vspace{10pt}\n");

            writer.write("\\begin{table}[htbp]\n");
            writer.write("\\centering\n");
            writer.write("\\begin{tabular}{l c c}\n");
            writer.write("\\toprule\n");
            writer.write("Service Name & Flow Delays (ms) & Max Delay (ms) \\\\\n");
            writer.write("\\midrule\n");

            // Write results for each service
            for (Map.Entry<String, List<Double>> entry : results.entrySet()) {
                String serviceName = entry.getKey();
                List<Double> delays = entry.getValue();
                double maxDelay = delays.stream().mapToDouble(Double::doubleValue).max().orElse(-1);
                String flowDelays = delays.stream()
                        .map(d -> String.format("%.2f", d))
                        .reduce((d1, d2) -> d1 + ", " + d2)
                        .orElse("N/A");

                writer.write(serviceName + " & " + flowDelays + " & " + String.format("%.2f", maxDelay) + " \\\\\n");
            }

            writer.write("\\bottomrule\n");
            writer.write("\\end{tabular}\n");
            writer.write("\\caption{Experiment results for Multiplexing: " + multiplexing + ", Analysis: " + ncAnalysisType + ", Arrival Bound: " + arrivalBoundMethod + "}\n");
            writer.write("\\end{table}\n");

            System.out.println("Results appended to " + combinedFileName);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Call this method at the end of all experiments to close the LaTeX document
    public void closeLatexFile() {
        try (FileWriter writer = new FileWriter(combinedFileName, true)) {
            writer.write("\\end{document}\n");
            System.out.println("LaTeX document closed.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }





    /**
     * Function used to remove all Flows from the current ServerGraph and
     * also remove all references made inside the SGService class
     */
    private void removeAllFlows() {
        for (Flow flow : this.serverGraph.getFlows()) {
            try {
                this.serverGraph.removeFlow(flow);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
        sgServices.forEach(SGService::resetFlowList);
    }

    /**
     * Helper function to remove all servers of the servergraph
     * and delete all references inside the edgeList
     */
    private void removeAllServers(){
        for (Server server : this.serverGraph.getServers()) {
            try {
                this.serverGraph.removeServer(server);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
        edgeList.forEach(Edge::resetServerList);
    }

    /**
     * Helper function used to reset the Server graph
     * and all references made inside the edgeList
     * and SGServices.
     * Preserves the initial definition of edges and services.
     */
    private void resetServerGraph(){
        removeAllFlows();
        removeAllServers();
        this.serverGraph = new ServerGraph();
    }

    /**
     * This function adds {@code nmbFlow} number of flows to the server graph (sg is modified in place).
     *
     * @param sg            Servergraph to add the flows to.
     * @param sgServiceList List of all available SGServices from which the flows shall be derived.
     * @param nmbFlow       number of flows which should be added. Use "-1" for all available flows.
     * @param fixedPrio     Use a fixed priority to choose the NC server where the flows shall be added.
     *                      Set to "null" if the SGS priority shall be used.
     */
    private void addFlowsToSG(ServerGraph sg, List<SGService> sgServiceList, int nmbFlow, FlowPriority fixedPrio) {
        // nmbFlow = -1 is used to add all available flows.
        if (nmbFlow == -1) {
            nmbFlow = Integer.MAX_VALUE;
        }
        // Add nmbFlow flows to the network (at most the available ones)
        int counter = 0;
        for (SGService service : sgServiceList) {
            // Create arrival curve with specified details
            ArrivalCurve arrival_curve = switch (experimentConfig.arrivalCurveType) {
                case TokenBucket ->
                        Curve.getFactory().createTokenBucket(service.getBitrate(), service.getBucket_size());
                case PeakArrivalRate -> Curve.getFactory().createPeakArrivalRate(service.getBitrate());
            };
            // Iterate over every field device - server combination (aka Path)
            for (int pathIdx = 0; pathIdx < service.getMultipath().size(); pathIdx++) {
                List<String> path = service.getMultipath().get(pathIdx);
                List<Server> dncPath = new ArrayList<>();
                List<String> edgeNodes = new ArrayList<>();
                // Find servers along path
                for (int i = 1; i < path.size(); i++) {  // Important: We start with the second item in the list!
                    edgeNodes.clear();
                    edgeNodes.add(path.get(i - 1));
                    edgeNodes.add(path.get(i));
                    // Add the found edge to the dncPath
                    if (fixedPrio == null) {
                        // The priority of the service shall be used
                        dncPath.add(findEdgebyNodes(edgeList, edgeNodes).getServer(service.getPriority()));
                    } else {
                        // A fixed value for the priority shall be used
                        dncPath.add(findEdgebyNodes(edgeList, edgeNodes).getServer(fixedPrio));
                    }
                }
                // Create flow and add it to the network
                try {
                    Flow flow = sg.addFlow(arrival_curve, dncPath);
                    service.addFlow(flow);
                    if (++counter >= nmbFlow) {
                        // Abort adding more flows
                        return;
                    }
                } catch (Exception e) {
                    //TODO: Exception Handling
                    throw new RuntimeException(e);
                }
            }
        }
    }

    /**
     * Test case which does a network calculus analysis after adding each flow.
     *
     * @param sg            ServerGraph which includes the servers and turns already
     * @param sgServiceList List of all available SGServices from which the flows shall be derived.
     */
    @SuppressWarnings("unused")
    private void testFlowAfterFlow(ServerGraph sg, List<SGService> sgServiceList) {
        // Get the total number of flows first
        int maxFlow = 0;
        for (SGService service : sgServiceList) {
            maxFlow += service.getMultipath().size();
        }

        for (int nmbFlow = 1; nmbFlow <= maxFlow; nmbFlow++) {
            addFlowsToSG(sg, sgServiceList, nmbFlow, FlowPriority.HIGH);
            // Safe the server graph
            this.serverGraph = sg;
            System.out.printf("%d Flows %n", sg.getFlows().size());

            calculateNCDelays();

            // Delete the flows
            for (Flow flow : sg.getFlows()) {
                try {
                    sg.removeFlow(flow);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
            for (SGService service : sgServiceList) {
                service.getFlows().clear();
            }
        }
    }


    /**
     * Test case which does a network calculus analysis after adding each flow.
     *
     * @param sg            ServerGraph which includes the servers and turns already
     * @param sgServiceList List of all available SGServices from which the flows shall be derived.
     */
    @SuppressWarnings("unused")
    private void testFlowPairs(ServerGraph sg, List<SGService> sgServiceList) {
        // Get the total number of flows first
        int maxFlow = 0;
        for (SGService service : sgServiceList) {
            maxFlow += service.getMultipath().size();
        }

        List<SGService> sgServiceListPre = new ArrayList<>();
        // Modify "max_depth" according to the test case you want to simulate
        recursiveCallFnc(sg, sgServiceList, sgServiceListPre, 1, 3);
    }

    /**
     * This function is used to test different combinations of flows. The function is meant as a recursive call, initialize the {@code curr_depth} with 1.
     *
     * @param sg                Disco server graph to use
     * @param sgServiceList     total list of SGS
     * @param servicesCumulated List of services already accumulated by previous recursive calls. Call with empty list as initial call.
     * @param curr_depth        current recursion depth. Initialize with 1 in initial call.
     * @param max_depth         maximal recursion depth aka number of flows per combination.
     */
    private void recursiveCallFnc(ServerGraph sg, List<SGService> sgServiceList, List<SGService> servicesCumulated, int curr_depth, int max_depth) {
        for (int serviceCntInner = 0; serviceCntInner < sgServiceList.size(); serviceCntInner++) {
            SGService serviceInner = sgServiceList.get(serviceCntInner);
            // Iterate over every flow in this service in outer loop
            for (int flowCntInner = 0; flowCntInner < serviceInner.getMultipath().size(); flowCntInner++) {
                List<String> pathInner = serviceInner.getMultipath().get(flowCntInner);
                // Add those two to the network and calculate
                List<List<String>> newPathListInner = new ArrayList<>();
                newPathListInner.add(pathInner);
                SGService serviceNewInner = new SGService(serviceInner.getName(), serviceInner.getServer(), serviceInner.getBucket_size(), serviceInner.getBitrate(), serviceInner.getDeadline(), newPathListInner, serviceInner.getPriority());
                // Add the two flows to the network
                List<SGService> sgServicesCompare = new ArrayList<>(servicesCumulated);
                sgServicesCompare.add(serviceNewInner);

                if (curr_depth >= max_depth) {
                    // Do the final computation
                    this.sgServices = sgServicesCompare;
                    addFlowsToSG(sg, sgServicesCompare, -1, FlowPriority.HIGH);
                    // Safe the server graph
                    this.serverGraph = sg;
                    System.out.printf("%d Flows %n", sg.getFlows().size());

                    calculateNCDelays();

                    // Delete the flows
                    for (Flow flow : sg.getFlows()) {
                        try {
                            sg.removeFlow(flow);
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                    for (SGService service : this.sgServices) {
                        service.getFlows().clear();
                    }
                } else {
                    recursiveCallFnc(sg, sgServiceList, sgServicesCompare, curr_depth + 1, max_depth);
                }
            }
        }
    }

    /**
     * Special test case for the presentation scenario, using the "SE" service, path "F23 - S1" and
     * the "LM" service, path "F12 - S2". Only adding those two flows, results in a stackoverflow.
     *
     * @param sg ServerGraph which includes the servers and turns already
     */
    @SuppressWarnings("unused")
    private void testBidirectionalFlow(ServerGraph sg) {
        {
            SGService service = sgServices.get(0);  // "SE" service
            // Create arrival curve with specified details
            ArrivalCurve arrival_curve = Curve.getFactory().createTokenBucket(service.getBitrate(), service.getBucket_size());

            int pathIdx = 4; // path "F23 - S1"
            List<String> path = service.getMultipath().get(pathIdx);
            List<Server> dncPath = new ArrayList<>();
            List<String> edgeNodes = new ArrayList<>();
            // Find servers along path
            for (int i = 1; i < path.size(); i++) {  // Important: We start with the second item in the list!
                edgeNodes.clear();
                edgeNodes.add(path.get(i - 1));
                edgeNodes.add(path.get(i));
                Collections.sort(edgeNodes);    // Important for comparison
                // Add the found edge to the dncPath
                dncPath.add(findEdgebyNodes(edgeList, edgeNodes).getServer());
            }
            // Create flow and add it to the network
            try {
                Flow flow = sg.addFlow(arrival_curve, dncPath);
                service.addFlow(flow);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
        {
            SGService service = sgServices.get(3);  // "LM" service
            // Create arrival curve with specified details
            ArrivalCurve arrival_curve = Curve.getFactory().createTokenBucket(service.getBitrate(), service.getBucket_size());

            int pathIdx = 0; // Path "F23 - S1"
            List<String> path = service.getMultipath().get(pathIdx);
            List<Server> dncPath = new ArrayList<>();
            List<String> edgeNodes = new ArrayList<>();
            // Find servers along path
            for (int i = 1; i < path.size(); i++) {  // Important: We start with the second item in the list!
                edgeNodes.clear();
                edgeNodes.add(path.get(i - 1));
                edgeNodes.add(path.get(i));
                Collections.sort(edgeNodes);    // Important for comparison
                // Add the found edge to the dncPath
                dncPath.add(findEdgebyNodes(edgeList, edgeNodes).getServer());
            }
            // Create flow and add it to the network
            try {
                Flow flow = sg.addFlow(arrival_curve, dncPath);
                service.addFlow(flow);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
        // Safe the server graph
        this.serverGraph = sg;
        System.out.printf("%d Flows %n", sg.getFlows().size());

        calculateNCDelays();

        // Delete the flows
        for (Flow flow : sg.getFlows()) {
            try {
                sg.removeFlow(flow);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
        for (SGService service : sgServices) {
            service.getFlows().clear();
        }
    }
}
